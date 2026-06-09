#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
好奇心模块验证实验 (v2)
验证：内在好奇心奖励能否驱动更快的环境探索？

修复：
  1. 好奇心奖励通过环境wrapper注入PPO
  2. 新奇区计数不随reset清零
  3. 两组都用PPO.learn()正常训练
"""

import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

try:
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as e:
    print(f"Missing: {e}"); sys.exit(1)

# ============================================================
# 前向动力学模型
# ============================================================
class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def get_error(self, state, action, next_state):
        with torch.no_grad():
            inp = torch.cat([state, action], dim=-1)
            pred = self.net(inp)
            error = self.loss_fn(pred, next_state).item()
        return error

    def update(self, state, action, next_state):
        inp = torch.cat([state, action], dim=-1)
        pred = self.net(inp)
        loss = self.loss_fn(pred, next_state)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ============================================================
# 好奇心环境 Wrapper（注入内在奖励）
# ============================================================
class CuriosityEnvWrapper(gym.Wrapper):
    """在每个step中计算好奇心奖励并加到环境返回的reward上"""
    def __init__(self, env, forward_model, curiosity_scale=0.3):
        super().__init__(env)
        self.fwd = forward_model
        self.scale = curiosity_scale
        self.last_obs = None
        self.last_action = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 计算好奇心奖励
        if self.last_obs is not None and self.last_action is not None:
            s = torch.FloatTensor(self.last_obs).unsqueeze(0)
            a = torch.FloatTensor(self.last_action).unsqueeze(0)
            ns = torch.FloatTensor(obs).unsqueeze(0)
            curiosity = self.fwd.get_error(s, a, ns)
            # 更新前向模型
            self.fwd.update(s, a, ns)
        else:
            curiosity = 0.0

        self.last_obs = obs.copy()
        self.last_action = action.copy()

        # 混合奖励
        total_reward = float(reward) + self.scale * curiosity
        info['curiosity_reward'] = curiosity
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs.copy()
        self.last_action = None
        return obs, info

# ============================================================
# 追踪 Callback：记录每步的 max_x
# ============================================================
class ExplorationTracker(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.max_x_history = []
        self.curiosity_values = []

    def _on_step(self):
        return True

# ============================================================
# 主实验
# ============================================================

def run(pretrain=500_000, explore=100_000):
    print("\n" + "="*60)
    print("  好奇心验证实验 v2")
    print("="*60)

    device = 'cpu'
    chunk = 4096  # 每chunk步训练一次
    n_chunks = explore // chunk

    # ===== 预训练 =====
    print("\n[Phase 1] 预训练 Ant...")
    env_pt = DummyVecEnv([lambda: gym.make('Ant-v5')])
    model_pt = PPO('MlpPolicy', env_pt, verbose=0,
                   n_steps=2048, batch_size=64, learning_rate=3e-4, device=device)
    model_pt.learn(total_timesteps=pretrain)
    pretrain_params = {n: p.data.clone() for n, p in model_pt.policy.named_parameters()}
    # 保存预训练模型供后续使用
    os.makedirs('results', exist_ok=True)
    torch.save(model_pt.policy.state_dict(), 'results/ant_pretrained.pt')
    env_pt.close()

    # 获取 state/action 维度
    tmp_env = gym.make('Ant-v5')
    state_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    # ===== 好奇心组 =====
    print("\n[Phase 2] 好奇心组...")
    fwd = ForwardModel(state_dim, action_dim).to(device)

    base_env_c = gym.make('Ant-v5')
    env_c_wrapped = CuriosityEnvWrapper(base_env_c, fwd, curiosity_scale=0.3)
    env_c = DummyVecEnv([lambda: env_c_wrapped])

    model_c = PPO('MlpPolicy', env_c, verbose=0,
                  n_steps=2048, batch_size=64, learning_rate=5e-5, device=device)
    model_c.policy.load_state_dict(model_pt.policy.state_dict())

    c_max_x_list = []
    c_curiosity_list = []

    for i in range(n_chunks):
        model_c.learn(total_timesteps=chunk, reset_num_timesteps=False)
        # 记录当前探索进度：跑一个评估episode
        obs, _ = env_c_wrapped.env.reset()
        max_x = 0
        total_cur = 0
        for _ in range(1000):
            action, _ = model_c.predict(obs, deterministic=False)
            obs, reward, done, _, _ = env_c_wrapped.env.step(action)
            max_x = max(max_x, env_c_wrapped.env.unwrapped.data.qpos[0])
            total_cur += float(reward)
            if done:
                break
        c_max_x_list.append(max_x)
        c_curiosity_list.append(total_cur)
        if i % max(1, n_chunks // 10) == 0:
            print(f"  好奇心组: chunk {i}/{n_chunks}, max_x: {max_x:.1f}")

    env_c.close()

    # ===== 对照组（无好奇心） =====
    print("\n[Phase 2] 对照组（无内在奖励）...")
    env_nc = DummyVecEnv([lambda: gym.make('Ant-v5')])
    model_nc = PPO('MlpPolicy', env_nc, verbose=0,
                   n_steps=2048, batch_size=64, learning_rate=5e-5, device=device)
    model_nc.policy.load_state_dict(model_pt.policy.state_dict())

    nc_max_x_list = []
    raw_env_nc = gym.make('Ant-v5').unwrapped

    for i in range(n_chunks):
        model_nc.learn(total_timesteps=chunk, reset_num_timesteps=False)
        obs, _ = raw_env_nc.reset()
        max_x = 0
        for _ in range(1000):
            action, _ = model_nc.predict(obs, deterministic=False)
            obs, _, done, _, _ = raw_env_nc.step(action)
            max_x = max(max_x, raw_env_nc.data.qpos[0])
            if done:
                break
        nc_max_x_list.append(max_x)
        if i % max(1, n_chunks // 10) == 0:
            print(f"  对照组: chunk {i}/{n_chunks}, max_x: {max_x:.1f}")

    raw_env_nc.close()
    env_nc.close()

    # 新奇区阈值 x > 3
    threshold = 3.0
    c_novel = sum(1 for x in c_max_x_list if x > threshold)
    nc_novel = sum(1 for x in nc_max_x_list if x > threshold)

    # 计算"首次进入新奇区"的chunk
    c_first = next((i for i, x in enumerate(c_max_x_list) if x > threshold), n_chunks)
    nc_first = next((i for i, x in enumerate(nc_max_x_list) if x > threshold), n_chunks)

    print("\n" + "="*60)
    print("  实验结果")
    print("="*60)
    print(f"  新奇区探索比例:  好奇心 {c_novel}/{n_chunks} = {c_novel/n_chunks*100:.1f}%")
    print(f"                     对照组 {nc_novel}/{n_chunks} = {nc_novel/n_chunks*100:.1f}%")
    print(f"  首次进入新奇区:   好奇心 chunk {c_first}, 对照组 chunk {nc_first}")
    if nc_first > 0:
        print(f"  好奇心加速倍数:   {nc_first/float(c_first):.1f}x" if c_first > 0 else "  ∞")
    print(f"  最终 max_x:        好奇心 {c_max_x_list[-1]:.1f}, 对照组 {nc_max_x_list[-1]:.1f}")

    # ===== 绘图 =====
    os.makedirs('results', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0d1117')

    # 图1: max_x 随 chunk 增长
    ax = axes[0, 0]
    chunks = list(range(n_chunks))
    ax.plot(chunks, c_max_x_list, color='#4ecdc4', linewidth=1.5, label='Curiosity')
    ax.plot(chunks, nc_max_x_list, color='#ff6b6b', linewidth=1.5, label='No Curiosity')
    ax.axhline(y=threshold, color='#ffe66d', linestyle='--', linewidth=1, alpha=0.7, label=f'Novel zone (x>{threshold})')
    ax.set_title('Max X Position Over Training', color='white', fontsize=12)
    ax.set_xlabel('Training Chunks', color='white')
    ax.set_ylabel('Max X Position', color='white')
    ax.legend(facecolor='#1a2332', edgecolor='#58a6ff', labelcolor='white', fontsize=9)
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')

    # 图2: 新奇区探索率对比
    ax = axes[0, 1]
    bars = ax.bar(['Curiosity', 'No Curiosity'],
                  [c_novel/n_chunks*100, nc_novel/n_chunks*100],
                  color=['#4ecdc4', '#ff6b6b'], edgecolor='white', linewidth=0.5, width=0.5)
    for bar, val in zip(bars, [c_novel/n_chunks*100, nc_novel/n_chunks*100]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')
    ax.set_title('Novel Zone Exploration Rate (%)', color='white', fontsize=12)
    ax.set_ylabel('% Chunks in Novel Zone', color='white')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')

    # 图3: 首次进入新奇区速度
    ax = axes[1, 0]
    bars = ax.bar(['Curiosity', 'No Curiosity'], [c_first, nc_first],
                  color=['#4ecdc4', '#ff6b6b'], edgecolor='white', linewidth=0.5, width=0.5)
    for bar, val in zip(bars, [c_first, nc_first]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val}', ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')
    ax.set_title('Chunks Until First Novel Zone Entry\n(lower = faster discovery)', color='white', fontsize=12)
    ax.set_ylabel('Chunks', color='white')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']: ax.spines[s].set_color('#333')

    # 图4: 总结
    ax = axes[1, 1]
    ax.axis('off')
    ax.set_facecolor('#0d1117')
    speedup = nc_first/float(c_first) if c_first > 0 else float('inf')
    summary = f"""
Curiosity Module Experiment
════════════════════════════════

Setup: MuJoCo Ant-v5
Novel zone: x > {threshold}
Pretraining: {pretrain//1000}K steps
Exploration: {explore//1000}K steps

Results
  Curiosity exploration rate:  {c_novel/n_chunks*100:.1f}%
  No-curiosity exploration:    {nc_novel/n_chunks*100:.1f}%
  First novel entry (cur):     chunk {c_first}
  First novel entry (no cur):  chunk {nc_first}
  Discovery speedup:           {speedup:.1f}x

Interpretation
  Intrinsic curiosity reward
  {'accelerates' if speedup > 1 else 'did not accelerate'}
  novel environment discovery
  by {speedup:.1f}x.
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            color='white', linespacing=1.3,
            bbox=dict(boxstyle='round', facecolor='#1a2332', edgecolor='#58a6ff', alpha=0.9))

    plt.tight_layout()
    fig.savefig('results/curiosity_experiment.png', dpi=200, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    with open('results/curiosity_data.json', 'w') as f:
        json.dump({
            'curiosity_rate': round(c_novel/n_chunks*100, 1),
            'control_rate': round(nc_novel/n_chunks*100, 1),
            'curiosity_first_entry': c_first,
            'control_first_entry': nc_first,
            'speedup': round(speedup, 1) if speedup != float('inf') else 'inf',
        }, f, indent=2)

    print("\n  完成！results/curiosity_experiment.png")

if __name__ == '__main__':
    run()
