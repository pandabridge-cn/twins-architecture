#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证附录定理：Fisher信息量与迁移保真度正相关
Naive vs EWC 对比：训练步数递增 → 零样本迁移
"""

import os, json, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

device = 'cpu'
checkpoints = [10000, 20000, 50000, 100000, 200000, 300000]

def make_env(task='A'):
    e = gym.make('Ant-v5')
    if task == 'B': e.unwrapped.model.opt.gravity[2] = -14.715
    return DummyVecEnv([lambda: e])

def ev(model, task='A', n=10):
    env = make_env(task); m, s = evaluate_policy(model, env, n); env.close(); return m, s

def sp(model):
    return {n: p.data.clone() for n, p in model.policy.named_parameters() if 'bias' not in n}

# EWC
class EWC:
    def __init__(self, model, old, lam=3000, n=150):
        self.old = old; self.lam = lam; self.f = {}
        e = gym.make('Ant-v5'); o, _ = e.reset()
        for _ in range(n):
            model.policy.zero_grad()
            t = torch.FloatTensor(o).unsqueeze(0)
            m = model.policy.action_net(model.policy.mlp_extractor.forward_actor(model.policy.features_extractor(t)))
            (-m.pow(2).sum()).backward(retain_graph=True)
            for nm, p in model.policy.named_parameters():
                if p.grad is not None:
                    self.f[nm] = self.f.get(nm, torch.zeros_like(p.data)) + p.grad.data**2
            o, _, tr, tu, _ = e.step(e.action_space.sample())
            if tr or tu: o, _ = e.reset()
        e.close()
        for k in self.f: self.f[k] /= n
    def pull(self, model, lr=0.001):
        for nm, p in model.policy.named_parameters():
            if nm in self.f and nm in self.old:
                p.data -= lr * self.lam * self.f[nm] * (p.data - self.old[nm])

print("="*60)
print("  Fidelity Verification: Naive vs EWC")
print("="*60)

# Train Naive
env_n = make_env('A')
model_n = PPO('MlpPolicy', env_n, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device=device, max_grad_norm=0.5)
naive = {'steps':[], 'a':[], 'b':[], 'ratio':[]}
steps_done = 0
for cp in checkpoints:
    model_n.learn(total_timesteps=cp - steps_done, reset_num_timesteps=False)
    steps_done = cp
    sa, _ = ev(model_n, 'A'); sb, _ = ev(model_n, 'B')
    naive['steps'].append(cp); naive['a'].append(sa); naive['b'].append(sb)
    naive['ratio'].append((sb/sa*100) if sa > 0 else 0)
    print(f"  Naive {cp//1000:4d}K | A:{sa:.0f} B:{sb:.0f} | {sb/sa*100:.1f}%")
env_n.close()

# Train EWC (start from same checkpoint, protect original)
print("---")
env_a = make_env('A')
model_a = PPO('MlpPolicy', env_a, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device=device, max_grad_norm=0.5)
model_a.learn(total_timesteps=checkpoints[0])
pa = sp(model_a)
ewc = EWC(model_a, pa)

ewc_data = {'steps':[], 'a':[], 'b':[], 'ratio':[]}
sa, _ = ev(model_a, 'A'); sb, _ = ev(model_a, 'B')
ewc_data['steps'].append(checkpoints[0]); ewc_data['a'].append(sa); ewc_data['b'].append(sb)
ewc_data['ratio'].append((sb/sa*100) if sa>0 else 0)
print(f"  EWC   {checkpoints[0]//1000:4d}K | A:{sa:.0f} B:{sb:.0f} | {sb/sa*100:.1f}%")

chunk = 5000
for cp in checkpoints[1:]:
    for _ in range(0, cp - checkpoints[checkpoints.index(cp)-1], chunk):
        model_a.learn(total_timesteps=chunk, reset_num_timesteps=False)
        ewc.pull(model_a)
    sa, _ = ev(model_a, 'A'); sb, _ = ev(model_a, 'B')
    ewc_data['steps'].append(cp); ewc_data['a'].append(sa); ewc_data['b'].append(sb)
    ewc_data['ratio'].append((sb/sa*100) if sa>0 else 0)
    print(f"  EWC   {cp//1000:4d}K | A:{sa:.0f} B:{sb:.0f} | {sb/sa*100:.1f}%")
env_a.close()

# ===== 绘图 =====
os.makedirs('results', exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0d1117')

sk = [s/1000 for s in checkpoints]
ax1.plot(sk, naive['ratio'], 's--', color='#ff6b6b', linewidth=2, markersize=8, label='Naive')
ax1.plot(sk, ewc_data['ratio'], 'o-', color='#4ecdc4', linewidth=2, markersize=8, label='EWC')
ax1.set_xlabel('Training Steps (K)', color='white', fontsize=12)
ax1.set_ylabel('Transfer Rate (%)', color='white', fontsize=12)
ax1.set_title('Fidelity vs Training: Naive vs EWC', color='white', fontsize=13, fontweight='bold')
ax1.legend(facecolor='#1a2332', edgecolor='#58a6ff', labelcolor='white', fontsize=11)
ax1.set_facecolor('#0d1117'); ax1.tick_params(colors='white')
ax1.grid(True, alpha=0.2, color='#58a6ff')
for s in ['top','right']: ax1.spines[s].set_visible(False)
for s in ['bottom','left']: ax1.spines[s].set_color('#333')

# Task A stability
ax2.plot(sk, naive['a'], 's--', color='#ff6b6b', linewidth=2, markersize=6, label='Naive Task A')
ax2.plot(sk, ewc_data['a'], 'o-', color='#4ecdc4', linewidth=2, markersize=6, label='EWC Task A')
ax2.set_xlabel('Training Steps (K)', color='white', fontsize=12)
ax2.set_ylabel('Task A Score', color='white', fontsize=12)
ax2.set_title('Task A Stability', color='white', fontsize=13, fontweight='bold')
ax2.legend(facecolor='#1a2332', edgecolor='#58a6ff', labelcolor='white', fontsize=11)
ax2.set_facecolor('#0d1117'); ax2.tick_params(colors='white')
ax2.grid(True, alpha=0.2, color='#58a6ff')
for s in ['top','right']: ax2.spines[s].set_visible(False)
for s in ['bottom','left']: ax2.spines[s].set_color('#333')

plt.tight_layout()
fig.savefig('results/fidelity_verification.png', dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()

with open('results/fidelity_data.json','w') as f:
    json.dump({'naive': naive, 'ewc': ewc_data}, f, indent=2)

print("\n  Final: Naive {:.1f}% vs EWC {:.1f}%".format(naive['ratio'][-1], ewc_data['ratio'][-1]))
print(f"  EWC advantage: {ewc_data['ratio'][-1] - naive['ratio'][-1]:.1f} percentage points")
print("  Saved: results/fidelity_verification.png")
