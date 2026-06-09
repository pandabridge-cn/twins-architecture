#!/usr/bin/env python3
"""Multi-novelty gradient: 5 gravity levels × 3 strategies, find negative transfer threshold."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

BASE_SEED = 42
torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED)

# ── Standard data ──
print("Collecting standard data...")
env = gym.make('HalfCheetah-v5')
o, _ = env.reset()
V_raw, F_raw = [], []
for i in range(3000):
    a = env.action_space.sample()
    o, _, t, tr, _ = env.step(a)
    V_raw.append(o[:9]); F_raw.append(a.copy())
    if t or tr: o, _ = env.reset()
env.close()
V = np.array(V_raw); F = np.array(F_raw)
Vm, Vs = V.mean(0), V.std(0)+1e-8
Fm, Fs = F.mean(0), F.std(0)+1e-8
Vstd = (V-Vm)/Vs; Fstd = (F-Fm)/Fs
Vt, Vte, Ft, Fte = train_test_split(Vstd, Fstd, test_size=0.3, random_state=42)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(9,128); self.l2 = nn.Linear(128,64); self.l3 = nn.Linear(64,6)
    def forward(self, x):
        x = torch.relu(self.l1(x)); x = torch.relu(self.l2(x)); return self.l3(x)

# Train reflex on standard
print("Training reflex on standard gravity...")
reflex = MLP()
opt = optim.Adam(reflex.parameters(), lr=0.001)
for _ in range(500):
    reflex.train(); opt.zero_grad()
    nn.MSELoss()(reflex(torch.FloatTensor(Vt)), torch.FloatTensor(Ft)).backward(); opt.step()
reflex.eval()
with torch.no_grad():
    r_std = r2_score(Fte, reflex(torch.FloatTensor(Vte)).numpy(), multioutput='uniform_average')
print(f"  Reflex R² on standard: {r_std:.3f}")

# ── Multi-gravity experiment ──
GRAVITIES = [9.81, 12.26, 14.72, 17.18, 19.62]  # 1.0×, 1.25×, 1.5×, 1.75×, 2.0×
LABELS = ['1.00×', '1.25×', '1.50×', '1.75×', '2.00×']
N_SEEDS = 5
N_EPOCHS = 100
N_SAMPLES = 2000

results = {}  # gravity -> {frozen_r2, scratch_r2s, warm_low_r2s, warm_anneal_r2s}

for gi, gravity in enumerate(GRAVITIES):
    label = LABELS[gi]
    print(f"\n{'='*50}")
    print(f"Gravity: {label} ({gravity} m/s²)")
    print(f"{'='*50}")

    # Collect data at this gravity
    env2 = gym.make('HalfCheetah-v5')
    env2.unwrapped.model.opt.gravity[2] = -gravity
    o, _ = env2.reset()
    Vnr, Fnr = [], []
    for i in range(N_SAMPLES):
        a = env2.action_space.sample()
        o, _, t, tr, _ = env2.step(a)
        Vnr.append(o[:9]); Fnr.append(a.copy())
        if t or tr: o, _ = env2.reset()
    env2.close()
    Vnov = (np.array(Vnr)-Vm)/Vs; Fnov = (np.array(Fnr)-Fm)/Fs
    Vnt = Vnov[:1500]; Fnt = Fnov[:1500]
    Vns = Vnov[1500:]; Fns = Fnov[1500:]
    Vnt_t = torch.FloatTensor(Vnt); Fnt_t = torch.FloatTensor(Fnt)
    Vns_t = torch.FloatTensor(Vns); Fns_t = torch.FloatTensor(Fns)

    # Frozen reflex
    with torch.no_grad():
        frozen_r2 = r2_score(Fns, reflex(Vns_t).numpy(), multioutput='uniform_average')
    print(f"  Frozen reflex R²: {frozen_r2:.3f}")

    scratch_finals, warm_low_finals, warm_anneal_finals = [], [], []
    scratch_curves, warm_low_curves, warm_anneal_curves = [], [], []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)

        # Scratch
        sc = MLP(); sc_opt = optim.Adam(sc.parameters(), lr=1e-3)
        # Warm-start low-LR
        wl = MLP(); wl.load_state_dict(reflex.state_dict()); wl_opt = optim.Adam(wl.parameters(), lr=1e-4)
        # Warm-start annealed
        wa = MLP(); wa.load_state_dict(reflex.state_dict()); wa_opt = optim.Adam(wa.parameters(), lr=1e-4)
        import math
        def get_lr(ep): return 1e-4 + (1e-3 - 1e-4) * (1 + math.cos(math.pi * ep / N_EPOCHS)) / 2

        sc_hist, wl_hist, wa_hist = [], [], []
        for ep in range(N_EPOCHS):
            for pg in wa_opt.param_groups: pg['lr'] = get_lr(ep)

            sc.train(); sc_opt.zero_grad()
            nn.MSELoss()(sc(Vnt_t), Fnt_t).backward(); sc_opt.step()

            wl.train(); wl_opt.zero_grad()
            nn.MSELoss()(wl(Vnt_t), Fnt_t).backward(); wl_opt.step()

            wa.train(); wa_opt.zero_grad()
            nn.MSELoss()(wa(Vnt_t), Fnt_t).backward(); wa_opt.step()

            sc.eval(); wl.eval(); wa.eval()
            with torch.no_grad():
                sc_hist.append(r2_score(Fns_t, sc(Vns_t).numpy(), multioutput='uniform_average'))
                wl_hist.append(r2_score(Fns_t, wl(Vns_t).numpy(), multioutput='uniform_average'))
                wa_hist.append(r2_score(Fns_t, wa(Vns_t).numpy(), multioutput='uniform_average'))

        scratch_finals.append(sc_hist[-1])
        warm_low_finals.append(wl_hist[-1])
        warm_anneal_finals.append(wa_hist[-1])
        scratch_curves.append(sc_hist)
        warm_low_curves.append(wl_hist)
        warm_anneal_curves.append(wa_hist)

    results[label] = {
        'frozen': frozen_r2,
        'scratch': (np.mean(scratch_finals), np.std(scratch_finals)),
        'warm_low': (np.mean(warm_low_finals), np.std(warm_low_finals)),
        'warm_anneal': (np.mean(warm_anneal_finals), np.std(warm_anneal_finals)),
        'scratch_curves': np.array(scratch_curves),
        'warm_low_curves': np.array(warm_low_curves),
        'warm_anneal_curves': np.array(warm_anneal_curves),
    }

    print(f"  Scratch:     {np.mean(scratch_finals):.3f} ± {np.std(scratch_finals):.3f}")
    print(f"  Warm(low):   {np.mean(warm_low_finals):.3f} ± {np.std(warm_low_finals):.3f}")
    print(f"  Warm(anneal):{np.mean(warm_anneal_finals):.3f} ± {np.std(warm_anneal_finals):.3f}")
    neg_transfer = frozen_r2 - np.mean(warm_low_finals)
    print(f"  Negative transfer (frozen - warm_low): {neg_transfer:+.3f}")

# ── Plot ──
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
grav_values = [1.0, 1.25, 1.5, 1.75, 2.0]

# (A) Frozen reflex R² across gravity levels
ax = axes[0, 0]
frozen_r2s = [results[l]['frozen'] for l in LABELS]
ax.plot(grav_values, frozen_r2s, 'o-', color='#333333', lw=2.5, markersize=10)
ax.fill_between(grav_values, frozen_r2s, alpha=0.1, color='#333333')
ax.set_xlabel('Gravity multiplier'); ax.set_ylabel('R²')
ax.set_title('Frozen Reflex Across Gravity Levels')
ax.set_ylim(0, 0.5)
for i, (g, r) in enumerate(zip(grav_values, frozen_r2s)):
    ax.text(g, r+0.02, f'{r:.3f}', ha='center', fontsize=9)

# (B) Final R² comparison across gravity levels
ax = axes[0, 1]
x = np.arange(len(LABELS)); w = 0.2
for i, (name, color, key) in enumerate([
    ('Frozen', '#333333', 'frozen'),
    ('Scratch', '#F44336', 'scratch'),
    ('Warm-low', '#2196F3', 'warm_low'),
    ('Warm-anneal', '#9C27B0', 'warm_anneal'),
]):
    offset = (i - 1.5) * w
    vals = [results[l][key] if key == 'frozen' else results[l][key][0] for l in LABELS]
    errs = [0 if key == 'frozen' else results[l][key][1] for l in LABELS]
    ax.bar(x + offset, vals, w, yerr=errs, color=color, edgecolor='white', capsize=3, label=name, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel('Final R²'); ax.set_title(f'Strategy Comparison ({N_EPOCHS} epochs)')
ax.legend(loc='lower left', fontsize=8)

# (C) Negative transfer magnitude
ax = axes[0, 2]
neg_transfers = [results[l]['frozen'] - results[l]['warm_low'][0] for l in LABELS]
colors_nt = ['#4CAF50' if nt < 0 else '#F44336' for nt in neg_transfers]
ax.bar(LABELS, neg_transfers, color=colors_nt, edgecolor='white')
ax.axhline(y=0, color='gray', ls='-', lw=0.5)
ax.set_ylabel('Negative Transfer\n(frozen − warm_low R²)')
ax.set_title('Negative Transfer Magnitude')
for i, nt in enumerate(neg_transfers):
    ax.text(i, nt + (0.02 if nt >= 0 else -0.04), f'{nt:+.3f}', ha='center', fontweight='bold', fontsize=10)

# (D) Scratch convergence curves across gravity levels
ax = axes[1, 0]
for li, label in enumerate(LABELS):
    c = plt.cm.Reds(0.3 + 0.7 * li / (len(LABELS)-1))
    m = results[label]['scratch_curves'].mean(0)
    s = results[label]['scratch_curves'].std(0)
    ax.fill_between(range(1, N_EPOCHS+1), m-s, m+s, alpha=0.08, color=c)
    ax.plot(range(1, N_EPOCHS+1), m, color=c, lw=2, label=label)
ax.set_xlabel('Epochs'); ax.set_ylabel('R²')
ax.set_title(f'Scratch Convergence ({N_SEEDS} seeds)')
ax.legend(fontsize=7)

# (E) Warm-low convergence curves
ax = axes[1, 1]
for li, label in enumerate(LABELS):
    c = plt.cm.Blues(0.3 + 0.7 * li / (len(LABELS)-1))
    m = results[label]['warm_low_curves'].mean(0)
    s = results[label]['warm_low_curves'].std(0)
    ax.fill_between(range(1, N_EPOCHS+1), m-s, m+s, alpha=0.08, color=c)
    ax.plot(range(1, N_EPOCHS+1), m, color=c, lw=2, label=label)
ax.set_xlabel('Epochs'); ax.set_ylabel('R²')
ax.set_title(f'Warm-start (lr=1e-4) Convergence')
ax.legend(fontsize=7)

# (F) Key insight: scratch advantage vs frozen
ax = axes[1, 2]
scratch_advantage = [results[l]['scratch'][0] - results[l]['frozen'] for l in LABELS]
cross_epochs = []
for li, label in enumerate(LABELS):
    sc_mean = results[label]['scratch_curves'].mean(0)
    frozen_val = results[label]['frozen']
    hits = np.where(sc_mean >= frozen_val)[0]
    cross_epochs.append(hits[0] + 1 if len(hits) > 0 else N_EPOCHS + 1)

ax2 = ax.twinx()
bars = ax.bar(LABELS, scratch_advantage, color=['#4CAF50' if sa > 0 else '#F44336' for sa in scratch_advantage], alpha=0.7)
ax.set_ylabel('Scratch − Frozen R²', color='#333')
ax2.plot(LABELS, cross_epochs, 'D-', color='#FF9800', lw=2, markersize=8)
ax2.set_ylabel('Epochs to surpass frozen', color='#FF9800')
ax.set_title('Scratch vs Frozen: Gap & Catch-up Time')
for i, (sa, ce) in enumerate(zip(scratch_advantage, cross_epochs)):
    ax.text(i, sa + (0.01 if sa >= 0 else -0.03), f'{sa:+.3f}', ha='center', fontsize=9)
    if ce <= N_EPOCHS:
        ax2.text(i, ce + 2, f'{ce}e', ha='center', fontsize=8, color='#FF9800')

plt.tight_layout()
plt.savefig('results/novelty_gradient.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/novelty_gradient.png")
print("Done!")
