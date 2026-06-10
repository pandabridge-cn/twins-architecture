#!/usr/bin/env python3
"""Twins Architecture: Push-Box Society v3 — Tabular Q-Learning.
Clean physical→social token migration. No neural nets, no noise."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
np.random.seed(42)

# ═══════════════════════════════
# TABULAR ENVIRONMENT + AGENT
# ═══════════════════════════════
class PushBox:
    """1D discrete. Box takes N pushes to reach goal. Single agent: N=8. Multi: N=5 (but needs 2 agents)."""
    def __init__(self, need_both=True):
        self.need_both = need_both  # if True, both must push for box to move
        self.max_steps = 30
        self.reset()

    def reset(self):
        self.box = 0
        self.both_pushed = False
        self.steps = 0
        return self.box

    def step(self, actions):
        """actions: list of ints, 0=idle, 1=push"""
        self.steps += 1
        moved = False

        if self.need_both:
            if all(a == 1 for a in actions):
                self.box += 1
                moved = True
        else:
            if any(a == 1 for a in actions):
                self.box += 1
                moved = True

        # Reward: box position (progress) + cooperation bonus
        r = self.box * 0.1
        if moved and self.need_both and len(actions) >= 2:
            r += 0.5  # cooperation bonus

        done = self.box >= 5 or self.steps >= self.max_steps
        if done and self.box >= 5:
            r += 2.0
        return self.box, r, done

class TabularQ:
    def __init__(self, n_states=6, n_actions=2, epsilon=0.2, alpha=0.1, gamma=0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.eps = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def act(self, s):
        if np.random.random() < self.eps:
            return np.random.randint(2)
        return np.argmax(self.Q[s])

    def learn(self, s, a, r, ns, done):
        target = r + (0 if done else self.gamma * np.max(self.Q[ns]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def greedy_act(self, s):
        return np.argmax(self.Q[s])

# ═══════════════════════════════
# EXPERIMENT
# ═══════════════════════════════
N_EP = 300; N_SEEDS = 10
print("=" * 60)
print("Twins Architecture: Push-Box Society v3 (Tabular)")
print("=" * 60)

# Phase 1: PHYSICAL — single agent, solo push
print("\n[Phase 1] Physical: single agent learns to push")
env_p = PushBox(need_both=False)
agent_p = TabularQ(n_states=6)
p_scores = []
for ep in range(N_EP):
    s = env_p.reset()
    total = 0; done = False
    while not done:
        a = agent_p.act(s); ns, r, done = env_p.step([a])
        agent_p.learn(s, a, r, ns, done); s = ns; total += r
    p_scores.append(total)
    if ep % 50 == 0: print(f"  ep {ep:3d}: reward={total:.2f}")
print(f"  → Physical Q-table:\n{np.round(agent_p.Q, 2)}")
phys_Q = agent_p.Q.copy()

# Phase 2: SOCIAL — two agents, need both to push
print(f"\n[Phase 2] Social: two agents must push TOGETHER ({N_SEEDS} seeds)")

# Group A: Physical-first (Twins) — agents know pushing gives reward
twins_all = []
for seed in range(N_SEEDS):
    np.random.seed(seed + 100)
    env = PushBox(need_both=True)
    a1 = TabularQ(n_states=6, epsilon=0.15); a2 = TabularQ(n_states=6, epsilon=0.15)
    a1.Q = phys_Q.copy(); a2.Q = phys_Q.copy()  # ← TOKEN MIGRATION
    scores = []
    for ep in range(N_EP):
        s = env.reset(); total = 0; done = False
        while not done:
            act1 = a1.act(s); act2 = a2.act(s)
            ns, r, done = env.step([act1, act2])
            a1.learn(s, act1, r, ns, done); a2.learn(s, act2, r, ns, done)
            s = ns; total += r
        scores.append(total)
    twins_all.append(scores)
    if seed % 3 == 0: print(f"  Twins seed {seed}: final={np.mean(scores[-50:]):.2f}")

# Group B: Social-only (Control) — agents start from zero
control_all = []
for seed in range(N_SEEDS):
    np.random.seed(seed + 200)
    env = PushBox(need_both=True)
    a1 = TabularQ(n_states=6, epsilon=0.2); a2 = TabularQ(n_states=6, epsilon=0.2)
    # NO physical Q-table!
    scores = []
    for ep in range(N_EP):
        s = env.reset(); total = 0; done = False
        while not done:
            act1 = a1.act(s); act2 = a2.act(s)
            ns, r, done = env.step([act1, act2])
            a1.learn(s, act1, r, ns, done); a2.learn(s, act2, r, ns, done)
            s = ns; total += r
        scores.append(total)
    control_all.append(scores)
    if seed % 3 == 0: print(f"  Control seed {seed}: final={np.mean(scores[-50:]):.2f}")

twins = np.array(twins_all); control = np.array(control_all)

# Stats
tw_f = twins[:, -50:].mean(); ct_f = control[:, -50:].mean()
tw_e = twins[:, :20].mean(); ct_e = control[:, :20].mean()

# Epochs to reach threshold
thr = 2.0
def ep2(c, t):
    e=[]; 
    for curve in c:
        h=np.where(np.array(curve)>=t)[0]
        e.append(h[0] if len(h)>0 else N_EP)
    return np.mean(e), np.std(e)
tw_et, tw_es = ep2(twins, thr); ct_et, ct_es = ep2(control, thr)

print(f"\n{'='*60}")
print(f"RESULTS ({N_SEEDS} seeds)")
print(f"{'='*60}")
print(f"  Physical final reward:      {np.mean(p_scores[-50:]):.2f}")
print(f"  Twins final reward:         {tw_f:.2f} ± {twins[:,-50:].mean(1).std():.2f}")
print(f"  Control final reward:       {ct_f:.2f} ± {control[:,-50:].mean(1).std():.2f}")
print(f"  Twins early advantage:      +{tw_e - ct_e:.2f} (eps 1-20)")
print(f"  Epochs to reward≥{thr}:  Twins={tw_et:.0f}±{tw_es:.0f}, Control={ct_et:.0f}±{ct_es:.0f}")
if ct_et < N_EP:
    print(f"  Speed-up:                   {ct_et/tw_et:.1f}×")
print(f"  Advantage consistent across seeds: {(twins[:,-50:].mean(1) > control[:,-50:].mean(1)).sum()}/{N_SEEDS}")
print(f"{'='*60}")

# PLOT
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (A) Physical
ax = axes[0]
ps = np.array(p_scores)
ax.plot(ps, alpha=0.3, color='#4CAF50', lw=0.5)
w=15; ax.plot(range(w-1,N_EP), np.convolve(ps,np.ones(w)/w,mode='valid'), color='#4CAF50',lw=2.5,label='Single Agent')
ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
ax.set_title('Phase 1: Physical Learning\n(Single agent learns to push)')
ax.legend()

# (B) Social
ax = axes[1]
for i in range(N_SEEDS):
    ax.plot(twins[i], alpha=0.12, color='#2196F3', lw=0.5)
    ax.plot(control[i], alpha=0.12, color='#F44336', lw=0.5)
tm=twins.mean(0); ts=twins.std(0); cm=control.mean(0); cs=control.std(0)
ax.fill_between(range(N_EP),tm-ts,tm+ts,alpha=0.12,color='#2196F3')
ax.fill_between(range(N_EP),cm-cs,cm+cs,alpha=0.12,color='#F44336')
ax.plot(tm,color='#2196F3',lw=2.5,label='Physical-first (Twins)')
ax.plot(cm,color='#F44336',lw=2.5,label='Social-only (Control)')
ax.axhline(y=thr,color='gray',ls='--',lw=0.5)
ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
ax.set_title(f'Phase 2: Cooperation ({N_SEEDS} seeds)')
ax.legend(loc='lower right')

# (C) Bar
ax = axes[2]
x=np.arange(3);wd=0.5
ax.bar(x[0],np.mean(p_scores[-50:]),wd,yerr=np.std(p_scores[-50:]),color='#4CAF50',edgecolor='white',capsize=6)
ax.bar(x[1],tw_f,wd,yerr=twins[:,-50:].mean(1).std(),color='#2196F3',edgecolor='white',capsize=6)
ax.bar(x[2],ct_f,wd,yerr=control[:,-50:].mean(1).std(),color='#F44336',edgecolor='white',capsize=6)
ax.set_xticks(x); ax.set_xticklabels(['Physical\n(single)','Twins\n(phys→social)','Control\n(social only)'])
ax.set_ylabel('Final Reward'); ax.set_title('Push-Box Society Results')
for bar,v in zip(ax.patches,[np.mean(p_scores[-50:]),tw_f,ct_f]):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.08,f'{v:.2f}',ha='center',fontweight='bold',fontsize=10)
ax.annotate(f'Twins\n+{tw_f-ct_f:.2f}',xy=(1,tw_f),xytext=(2.1,max(tw_f,ct_f)+0.3),
            arrowprops=dict(arrowstyle='->',color='#2196F3',lw=2),fontsize=10,color='#2196F3',fontweight='bold')

plt.tight_layout()
plt.savefig('results/pushbox_society.png', dpi=150, bbox_inches='tight')
print("\nSaved: results/pushbox_society.png")
