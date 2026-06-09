#!/usr/bin/env python3
"""Four anti-forgetting methods on Ant-v5: Naive, EWC, SI, MAS. CPU ~6-8h."""

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
TRAIN = 300_000
CHUNK = 5000

def make_env(task='A'):
    e = gym.make('Ant-v5')
    if task == 'B': e.unwrapped.model.opt.gravity[2] = -14.715
    elif task == 'C':
        for i in range(e.unwrapped.model.geom_friction.shape[0]):
            e.unwrapped.model.geom_friction[i, 0] = 0.3
    return DummyVecEnv([lambda: e])

def ev(model, task='A', n=20):
    env = make_env(task); m, s = evaluate_policy(model, env, n); env.close(); return m, s

def sp(model):
    return {n: p.data.clone() for n, p in model.policy.named_parameters() if 'bias' not in n}

class EWC:
    def __init__(self, model, old_p, lam=5000, n=200):
        self.old = old_p; self.lam = lam; self.f = {}
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

class SI:
    def __init__(self, old_p, xi=1.0):
        self.old = old_p; self.xi = xi
        self.omega = {n: torch.zeros_like(p) for n, p in old_p.items()}
        self.prev = None
    def snapshot(self, model):
        self.prev = {n: p.data.clone() for n, p in model.policy.named_parameters() if n in self.old}
    def update(self, model):
        if self.prev is None: self.snapshot(model); return
        for n, p in model.policy.named_parameters():
            if n in self.prev and n in self.old:
                self.omega[n] += torch.abs(p.data - self.prev[n])
                self.prev[n] = p.data.clone()
    def pull(self, model, lr=0.001):
        for n, p in model.policy.named_parameters():
            if n in self.omega and n in self.old:
                p.data -= lr * self.xi * self.omega[n] * (p.data - self.old[n])

class MAS:
    def __init__(self, model, old_p, lam=1, n=200):
        self.old = old_p; self.lam = lam; self.imp = {}
        e = gym.make('Ant-v5'); o, _ = e.reset()
        for _ in range(n):
            model.policy.zero_grad()
            t = torch.FloatTensor(o).unsqueeze(0)
            m = model.policy.action_net(model.policy.mlp_extractor.forward_actor(model.policy.features_extractor(t)))
            m.pow(2).sum().backward(retain_graph=True)
            for nm, p in model.policy.named_parameters():
                if p.grad is not None:
                    self.imp[nm] = self.imp.get(nm, torch.zeros_like(p.data)) + torch.abs(p.grad.data)
            o, _, tr, tu, _ = e.step(e.action_space.sample())
            if tr or tu: o, _ = e.reset()
        e.close()
        for k in self.imp: self.imp[k] /= n
    def pull(self, model, lr=0.001):
        for nm, p in model.policy.named_parameters():
            if nm in self.imp and nm in self.old:
                p.data -= lr * self.lam * self.imp[nm] * (p.data - self.old[nm])

def run():
    print("\n" + "="*60)
    print("  Baseline: Naive vs EWC vs SI vs MAS (300K steps/task)")
    print("="*60)

    print("\n[Task A] Training...")
    env_a = make_env('A')
    ma = PPO('MlpPolicy', env_a, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device=device, max_grad_norm=0.5)
    ma.learn(total_timesteps=TRAIN)
    pa = sp(ma)
    base, _ = ev(ma, 'A')
    print(f"  Baseline: {base:.0f}")
    env_a.close()

    ewc = EWC(ma, pa); mas = MAS(ma, pa)
    methods = ['naive', 'ewc', 'si', 'mas']
    results = {}

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        si_obj = SI(pa) if method == 'si' else None
        if si_obj: si_obj.snapshot(ma)

        # Task B
        eb = make_env('B')
        mb = PPO('MlpPolicy', eb, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device=device, max_grad_norm=0.5)
        mb.policy.load_state_dict(ma.policy.state_dict())
        for i in range(0, TRAIN, CHUNK):
            mb.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
            if method == 'ewc': ewc.pull(mb)
            elif method == 'mas': mas.pull(mb)
            elif method == 'si': si_obj.update(mb); si_obj.pull(mb)
        sb, _ = ev(mb, 'A')
        print(f"  After B: {sb/base*100:.1f}%")
        eb.close()

        # Task C
        ec = make_env('C')
        mc = PPO('MlpPolicy', ec, verbose=0, n_steps=2048, batch_size=64, learning_rate=3e-4, device=device, max_grad_norm=0.5)
        mc.policy.load_state_dict(mb.policy.state_dict())
        for i in range(0, TRAIN, CHUNK):
            mc.learn(total_timesteps=CHUNK, reset_num_timesteps=False)
            if method == 'ewc': ewc.pull(mc)
            elif method == 'mas': mas.pull(mc)
            elif method == 'si': si_obj.update(mc); si_obj.pull(mc)
        sc, _ = ev(mc, 'A')
        print(f"  After C: {sc/base*100:.1f}%")
        ec.close()
        results[method] = {'b': sb/base*100, 'c': sc/base*100}

    os.makedirs('results', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d1117')
    names = ['Naive', 'EWC', 'SI', 'MAS']
    colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a855f7']
    x = np.arange(4)
    for ax, key, title in [(ax1, 'b', 'After 1 Task Switch'), (ax2, 'c', 'After 2 Task Switches')]:
        vals = [results[m][key] for m in methods]
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{val:.1f}%',
                    ha='center', color='white', fontsize=11, fontweight='bold')
        ax.set_title(title, color='white', fontsize=13)
        ax.set_xticks(x); ax.set_xticklabels(names, color='white', fontsize=11)
        ax.axhline(y=80, color='#888', ls='--', alpha=0.5)
        ax.set_facecolor('#0d1117'); ax.tick_params(colors='white')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['bottom','left']: ax.spines[s].set_color('#333')
    plt.tight_layout()
    fig.savefig('results/method_comparison.png', dpi=200, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    with open('results/method_comparison.json','w') as f:
        json.dump({'baseline': float(base), 'results': {m: {'after_b': float(v['b']), 'after_c': float(v['c'])} for m, v in results.items()}}, f, indent=2)
    print("\n" + "="*60 + "\n  DONE!")
    for m in methods: print(f"  {m:6s}: {results[m]['c']:.1f}%")
    print("="*60)

if __name__ == '__main__': run()
