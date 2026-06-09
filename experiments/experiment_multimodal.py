#!/usr/bin/env python3
"""Multimodal alignment: 5-seed with CORRECT time-shifting."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})

BASE_SEED = 42; N_SEEDS = 5
torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED)

print("Collecting multimodal data...")
env = gym.make('HalfCheetah-v5')
o, _ = env.reset()
V_raw, F_raw, A_raw = [], [], []
for i in range(10000):
    a = env.action_space.sample()
    o, _, t, tr, _ = env.step(a)
    V_raw.append(o[:9])
    F_raw.append(a.copy())
    A_raw.append([np.linalg.norm(o[8:11]), np.linalg.norm(o[11:14]), np.linalg.norm(o[14:17])])
    if t or tr: o, _ = env.reset()
env.close()

V = np.array(V_raw); F = np.array(F_raw); A = np.array(A_raw)
Vm, Vs = V.mean(0), V.std(0)+1e-8
Fm, Fs = F.mean(0), F.std(0)+1e-8
Am, As = A.mean(0), A.std(0)+1e-8
Vn = (V-Vm)/Vs; Fn = (F-Fm)/Fs; An = (A-Am)/As

shift = 20
# Aligned: same time index
V_aligned = Vn[:-shift]; F_aligned = Fn[:-shift]; A_aligned = An[:-shift]
# Shifted: V at t+shift, F at t (time misaligned!)
V_shifted = Vn[shift:]; F_shifted = Fn[:-shift]; A_shifted = An[:-shift]

DIRECTIONS = [
    ('V→F', 9, 6, V_aligned, F_aligned, V_shifted, F_shifted),
    ('V→A', 9, 3, V_aligned, A_aligned, V_shifted, A_shifted),
    ('F→V', 6, 9, F_aligned, V_aligned, F_shifted, V_shifted),
    ('F→A', 6, 3, F_aligned, A_aligned, F_shifted, A_shifted),
    ('A→V', 3, 9, A_aligned, V_aligned, A_shifted, V_shifted),
    ('A→F', 3, 6, A_aligned, F_aligned, A_shifted, F_shifted),
]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,out_dim))
    def forward(self, x): return self.net(x)

print(f"\n{N_SEEDS}-seed multimodal alignment (correctly time-shifted)\n")
all_results = {}

for name, in_dim, out_dim, Xa, Ya, Xs, Ys in DIRECTIONS:
    aligned_r2s, shifted_r2s = [], []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        model = MLP(in_dim, out_dim)
        opt = optim.Adam(model.parameters(), lr=0.001)
        Xa_t = torch.FloatTensor(Xa); Ya_t = torch.FloatTensor(Ya)
        Xs_t = torch.FloatTensor(Xs); Ys_t = torch.FloatTensor(Ys)
        for _ in range(300):
            model.train(); opt.zero_grad()
            nn.MSELoss()(model(Xa_t), Ya_t).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            aligned_r2s.append(r2_score(Ya, model(Xa_t).numpy(), multioutput='uniform_average'))
            shifted_r2s.append(r2_score(Ys, model(Xs_t).numpy(), multioutput='uniform_average'))
    all_results[name] = {
        'aligned_mean': np.mean(aligned_r2s), 'aligned_std': np.std(aligned_r2s),
        'shifted_mean': np.mean(shifted_r2s), 'shifted_std': np.std(shifted_r2s),
    }
    print(f"  {name}: aligned={np.mean(aligned_r2s):.3f}±{np.std(aligned_r2s):.3f}, shifted={np.mean(shifted_r2s):.3f}±{np.std(shifted_r2s):.3f}")

# Print aggregate
ali_vals = [all_results[d]['aligned_mean'] for d in [n for n,_,_,_,_,_,_ in DIRECTIONS]]
shi_vals = [all_results[d]['shifted_mean'] for d in [n for n,_,_,_,_,_,_ in DIRECTIONS]]
print(f"\n  Aggregate aligned: {np.mean(ali_vals):.3f}±{np.std(ali_vals):.3f}")
print(f"  Aggregate shifted: {np.mean(shi_vals):.3f}±{np.std(shi_vals):.3f}")
print(f"  Δ = {np.mean(ali_vals)-np.mean(shi_vals):.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ax = axes[0]
x = np.arange(6); w = 0.35
names_short = ['V→F','V→A','F→V','F→A','A→V','A→F']
for i, (label, key, color) in enumerate([
    ('Aligned', 'aligned', '#2196F3'),
    ('Shifted (+20)', 'shifted', '#F44336'),
]):
    vals = [all_results[d][f'{key}_mean'] for d in names_short]
    errs = [all_results[d][f'{key}_std'] for d in names_short]
    ax.bar(x + i*w, vals, w, yerr=errs, color=color, edgecolor='white', capsize=4, alpha=0.85, label=label)
ax.axhline(y=0, color='gray', ls='--', lw=0.5)
ax.set_xticks(x + w/2); ax.set_xticklabels(names_short)
ax.set_ylabel('R²'); ax.set_title(f'Cross-Modal Prediction ({N_SEEDS} seeds, ±1σ)')
ax.legend()

ax = axes[1]
ax.bar(['Aligned', 'Shifted (+20 steps)'], [np.mean(ali_vals), np.mean(shi_vals)],
       yerr=[np.std(ali_vals), np.std(shi_vals)],
       color=['#2196F3', '#F44336'], edgecolor='white', width=0.5, capsize=8)
ax.axhline(y=0, color='gray', ls='--', lw=0.5)
ax.set_ylabel('Mean R²'); ax.set_title('Aggregate')
for i, (v, s) in enumerate([(np.mean(ali_vals), np.std(ali_vals)), (np.mean(shi_vals), np.std(shi_vals))]):
    ax.text(i, v + s + 0.02, f'{v:.3f}±{s:.3f}', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('results/multimodal_prediction.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/multimodal_prediction.png")
