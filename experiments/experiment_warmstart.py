#!/usr/bin/env python3
"""Warm-start v5: full LR + annealing. Test if pre-trained weights help w/o LR constraint."""
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

BASE_SEED = 42
torch.manual_seed(BASE_SEED); np.random.seed(BASE_SEED)

# Data
env = gym.make('HalfCheetah-v5')
o, _ = env.reset()
V_raw, F_raw = [], []
for i in range(2000):
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

env2 = gym.make('HalfCheetah-v5')
env2.unwrapped.model.opt.gravity[2] = -14.715
o, _ = env2.reset()
Vnr, Fnr = [], []
for i in range(2000):
    a = env2.action_space.sample()
    o, _, t, tr, _ = env2.step(a)
    Vnr.append(o[:9]); Fnr.append(a.copy())
    if t or tr: o, _ = env2.reset()
env2.close()
Vnov = (np.array(Vnr)-Vm)/Vs; Fnov = (np.array(Fnr)-Fm)/Fs
Vnt, Vns = Vnov[:1500], Vnov[1500:]
Fnt, Fns = Fnov[:1500], Fnov[1500:]

Vnt_t = torch.FloatTensor(Vnt); Fnt_t = torch.FloatTensor(Fnt)
Vns_t = torch.FloatTensor(Vns); Fns_t = torch.FloatTensor(Fns)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(9, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 6)
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

# Train reflex
reflex = MLP()
opt = optim.Adam(reflex.parameters(), lr=0.001)
for _ in range(500):
    reflex.train(); opt.zero_grad()
    nn.MSELoss()(reflex(torch.FloatTensor(Vt)), torch.FloatTensor(Ft)).backward()
    opt.step()
reflex.eval()
with torch.no_grad():
    r_reflex_nov = r2_score(Fns, reflex(Vns_t).numpy(), multioutput='uniform_average')
print(f"Reflex R² on novel: {r_reflex_nov:.3f}")

N_SEEDS, N_EPOCHS = 5, 100

# Three strategies
# (a) Warm-start, same LR as scratch (1e-3) — test if pretrained weights help unconstrained
# (b) Warm-start, annealed LR (1e-4 → 1e-3) — ramp up exploration
# (c) Scratch, LR=1e-3 — baseline

print(f"\n{N_SEEDS} seeds, {N_EPOCHS} epochs")

curves_warm_full = []   # (a)
curves_warm_anneal = [] # (b) 
curves_scratch = []     # (c)

for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)

    # (a) Warm-start with full LR
    wf = MLP(); wf.load_state_dict(reflex.state_dict())
    wf_opt = optim.Adam(wf.parameters(), lr=1e-3)  # SAME as scratch

    # (b) Warm-start with annealed LR
    wa = MLP(); wa.load_state_dict(reflex.state_dict())
    wa_opt = optim.Adam(wa.parameters(), lr=1e-4)
    # Cosine annealing: lr(epoch) = lr_min + (lr_max - lr_min) * (1 + cos(pi * epoch / N)) / 2
    import math
    lr_min, lr_max = 1e-4, 1e-3
    def get_lr(epoch):
        return lr_min + (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / N_EPOCHS)) / 2

    # (c) Scratch
    sc = MLP()
    sc_opt = optim.Adam(sc.parameters(), lr=1e-3)

    wf_hist, wa_hist, sc_hist = [], [], []

    for ep in range(N_EPOCHS):
        # Update annealed LR
        for pg in wa_opt.param_groups:
            pg['lr'] = get_lr(ep)

        # (a) Warm full-LR step
        wf.train(); wf_opt.zero_grad()
        nn.MSELoss()(wf(Vnt_t), Fnt_t).backward(); wf_opt.step()

        # (b) Warm annealed step
        wa.train(); wa_opt.zero_grad()
        nn.MSELoss()(wa(Vnt_t), Fnt_t).backward(); wa_opt.step()

        # (c) Scratch step
        sc.train(); sc_opt.zero_grad()
        nn.MSELoss()(sc(Vnt_t), Fnt_t).backward(); sc_opt.step()

        wf.eval(); wa.eval(); sc.eval()
        with torch.no_grad():
            wf_hist.append(r2_score(Fns_t, wf(Vns_t).numpy(), multioutput='uniform_average'))
            wa_hist.append(r2_score(Fns_t, wa(Vns_t).numpy(), multioutput='uniform_average'))
            sc_hist.append(r2_score(Fns_t, sc(Vns_t).numpy(), multioutput='uniform_average'))

    curves_warm_full.append(wf_hist)
    curves_warm_anneal.append(wa_hist)
    curves_scratch.append(sc_hist)

    print(f"  Seed {seed}: warm_full={wf_hist[-1]:.3f}, warm_anneal={wa_hist[-1]:.3f}, scratch={sc_hist[-1]:.3f}")

curves_warm_full = np.array(curves_warm_full)
curves_warm_anneal = np.array(curves_warm_anneal)
curves_scratch = np.array(curves_scratch)

# Stats
for name, curves in [("Warm-full-LR", curves_warm_full),
                       ("Warm-annealed", curves_warm_anneal),
                       ("Scratch", curves_scratch)]:
    m, s = curves[:, -1].mean(), curves[:, -1].std()
    start = curves[:, 0].mean()
    print(f"  {name}: start={start:.3f}, final={m:.3f}±{s:.3f}")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_arr = np.arange(1, N_EPOCHS + 1)

# (a) Full convergence
ax = axes[0]
colors = {'Warm-full-LR': '#FF9800', 'Warm-annealed': '#9C27B0', 'Scratch': '#F44336'}
for label, curves in [("Warm-full-LR", curves_warm_full),
                        ("Warm-annealed", curves_warm_anneal),
                        ("Scratch", curves_scratch)]:
    m, s = curves.mean(0), curves.std(0)
    ax.fill_between(epochs_arr, m-s, m+s, alpha=0.12, color=colors[label])
    ax.plot(epochs_arr, m, color=colors[label], lw=2, label=label)
ax.axhline(y=r_reflex_nov, color='gray', ls='--', alpha=0.4)
ax.set_xlabel('Epochs'); ax.set_ylabel('R²')
ax.set_title(f'Convergence ({N_SEEDS} seeds)')
ax.legend()

# (b) Zoom: first 30 epochs
ax = axes[1]
for label, curves in [("Warm-full-LR", curves_warm_full),
                        ("Warm-annealed", curves_warm_anneal),
                        ("Scratch", curves_scratch)]:
    m, s = curves.mean(0), curves.std(0)
    ax.fill_between(epochs_arr[:30], (m-s)[:30], (m+s)[:30], alpha=0.12, color=colors[label])
    ax.plot(epochs_arr[:30], m[:30], color=colors[label], lw=2, label=label)
ax.axhline(y=r_reflex_nov, color='gray', ls='--', alpha=0.4)
ax.set_xlabel('Epochs'); ax.set_ylabel('R²')
ax.set_title('First 30 epochs (zoom)')
ax.legend()

# (c) Final comparison
ax = axes[2]
x = np.arange(3); wd = 0.25
data = [
    (curves_warm_full[:, -1], 'Warm\nfull-LR', '#FF9800'),
    (curves_warm_anneal[:, -1], 'Warm\nannealed', '#9C27B0'),
    (curves_scratch[:, -1], 'Scratch', '#F44336'),
]
for i, (vals, label, color) in enumerate(data):
    ax.bar(i, vals.mean(), wd, yerr=vals.std(), color=color, edgecolor='white', capsize=8)
    ax.text(i, vals.mean()+vals.std()+0.01, f'{vals.mean():.3f}±{vals.std():.3f}',
            ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([d[1] for d in data])
ax.set_ylabel('Final R²')
ax.set_title('After 100 epochs')

plt.tight_layout()
plt.savefig('results/warmstart_validation.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/warmstart_validation.png")
