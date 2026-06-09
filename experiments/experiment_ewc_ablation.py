#!/usr/bin/env python3
"""EWC lambda ablation on Ant-v5. Full 300K steps × 3 tasks × 6 lambda values.
Saves checkpoint every lambda so crash doesn't lose everything.
Estimated runtime on i7-11390H: ~4-6 hours."""

import os, json, warnings, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# ── Config ──
TRAIN_STEPS = 300_000
LAMBDA_VALUES = [1, 10, 100, 500, 1000, 5000, 10000]
N_REPLAY_SAMPLES = 200  # samples for Fisher estimation
EVAL_EPISODES = 20
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, 'ewc_ablation_checkpoint.json')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'ewc_ablation_results.json')

print("="*60)
print("EWC λ Ablation on Ant-v5")
print(f"λ values: {LAMBDA_VALUES}")
print(f"Steps per task: {TRAIN_STEPS:,}")
print(f"Estimated time: {len(LAMBDA_VALUES)*1:.0f}-{len(LAMBDA_VALUES)*1.5:.0f} hours")
print("="*60)

# ── Environment factory ──
def make_env(gravity_mult=1.0, friction_mult=1.0):
    def _init():
        env = gym.make('Ant-v5')
        env.unwrapped.model.opt.gravity[2] = -9.81 * gravity_mult
        if friction_mult != 1.0:
            for i in range(env.unwrapped.model.geom_friction.shape[0]):
                env.unwrapped.model.geom_friction[i, 0] = 1.0 * friction_mult
        return env
    return _init

def evaluate(model, env_fn, n_episodes=EVAL_EPISODES):
    env = DummyVecEnv([env_fn])
    mean, std = evaluate_policy(model, env, n_episodes)
    env.close()
    return mean

# ── EWC implementation (same as experiment_baselines.py) ──
class EWCCallback:
    def __init__(self, model, old_params, lam, n_samples=N_REPLAY_SAMPLES):
        self.old = old_params
        self.lam = lam
        self.fisher = {}
        # Estimate Fisher information on Task A data
        env = gym.make('Ant-v5')
        obs, _ = env.reset()
        for _ in range(n_samples):
            model.policy.zero_grad()
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            # Use action entropy as proxy for Fisher (same approach as original)
            features = model.policy.features_extractor(obs_t)
            latent = model.policy.mlp_extractor.forward_actor(features)
            actions = model.policy.action_net(latent)
            (-actions.pow(2).sum()).backward(retain_graph=True)
            for name, param in model.policy.named_parameters():
                if param.grad is not None:
                    self.fisher[name] = self.fisher.get(name, torch.zeros_like(param.data)) + param.grad.data ** 2
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()

    def __call__(self, _locals, _globals):
        # Apply EWC penalty during training
        ewc_loss = 0.0
        for name, param in _locals['self'].policy.named_parameters():
            if name in self.old and name in self.fisher:
                ewc_loss += (self.fisher[name] * (param - self.old[name]) ** 2).sum()
        # Inject into PPO loss
        if hasattr(_locals['self'], 'ewc_lambda'):
            _locals['self'].ewc_loss = self.lam * ewc_loss

# Monkey-patch PPO to apply EWC loss
import stable_baselines3.ppo.ppo as ppo_module
original_train = ppo_module.PPO.train

def train_with_ewc(self):
    if hasattr(self, 'ewc_callback') and hasattr(self, 'ewc_loss'):
        self.ewc_callback(self, None)
        # Add EWC loss to policy loss
        self.policy.optimizer.zero_grad()
        # The original train() handles this — we need a different approach
    return original_train(self)

# ── Simpler approach: manually inject EWC loss every N steps ──
# We'll use a custom callback approach instead

from stable_baselines3.common.callbacks import BaseCallback

class EWCCallback2(BaseCallback):
    def __init__(self, old_params, lam, fisher, verbose=0):
        super().__init__(verbose)
        self.old = old_params
        self.lam = lam
        self.fisher = fisher

    def _on_step(self):
        # Compute EWC penalty and add to model
        ewc_loss = 0.0
        for name, param in self.model.policy.named_parameters():
            if name in self.old and name in self.fisher:
                ewc_loss += (self.fisher[name] * (param - self.old[name]) ** 2).sum()

        # Add EWC loss to the model's loss by setting a temporary attribute
        # This works because SB3's PPO adds the loss in the optimizer step
        if not hasattr(self.model, '_ewc_losses'):
            self.model._ewc_losses = []
        self.model._ewc_losses.append(float(self.lam * ewc_loss))
        return True

def train_task_with_ewc(model, env_fn, total_steps, old_params, fisher, lam):
    """Train model on new task with EWC regularization.
    Strategy: train normally, then after each rollout, manually apply EWC gradient step."""
    env = DummyVecEnv([env_fn])

    # Save initial params
    model.set_env(env)

    # Train with a modified PPO that includes EWC
    # We override the loss function
    original_train_fn = model.__class__.train

    class EWCWrapper:
        def __init__(self, model, old, fisher, lam):
            self.model = model
            self.old = old
            self.fisher = fisher
            self.lam = lam

        def __enter__(self):
            self.orig_loss = self.model.policy.loss_fn if hasattr(self.model.policy, 'loss_fn') else None
            return self

        def __exit__(self, *args):
            pass

    # Simpler: just train without callback, then apply EWC penalty manually every 1000 steps
    # Actually, let's use the simplest possible approach: just run model.learn()

    model.learn(total_timesteps=total_steps, progress_bar=False)

    # Apply EWC penalty manually as additional gradient steps
    model.policy.train()
    optimizer = model.policy.optimizer
    for _ in range(100):  # 100 extra steps of EWC penalty
        optimizer.zero_grad()
        ewc_loss = torch.tensor(0.0)
        for name, param in model.policy.named_parameters():
            if name in old and name in fisher:
                ewc_loss = ewc_loss + (fisher[name] * (param - old[name]) ** 2).sum()
        (self.lam * ewc_loss).backward()
        optimizer.step()

    env.close()
    return model

# This approach is getting too complex. Let me use a much simpler method:
# Just periodically apply EWC as an additional gradient step.

def train_with_ewc_simple(model, env_fn, total_steps, old_params, fisher, lam, eval_env_fn=None):
    """Train with EWC by adding penalty gradient steps periodically."""
    env = DummyVecEnv([env_fn])

    # We'll manually step through training and inject EWC
    n_steps = 2048  # PPO default
    n_epochs = 10   # PPO default
    iterations = total_steps // n_steps
    ewc_freq = max(1, iterations // 50)  # Apply EWC penalty ~50 times during training

    model.set_env(env)
    optimizer = model.policy.optimizer
    model.policy.train()

    for it in range(iterations):
        # Collect rollout and train normally
        model.learn(total_timesteps=n_steps, reset_num_timesteps=False, progress_bar=False)

        # Apply EWC penalty periodically
        if it % ewc_freq == 0:
            optimizer.zero_grad()
            ewc_loss = torch.tensor(0.0)
            for name, param in model.policy.named_parameters():
                if name in old_params and name in fisher:
                    ewc_loss = ewc_loss + (fisher[name] * (param - old_params[name]) ** 2).sum()
            (lam * ewc_loss).backward()
            optimizer.step()

    env.close()
    return model

# ── Load or init checkpoint ──
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)
    completed_lambdas = set(ckpt.get('completed', []))
    task_a_score = ckpt.get('task_a_score', None)
    print(f"Resuming from checkpoint. Completed: {completed_lambdas}")
else:
    completed_lambdas = set()
    task_a_score = None
    ckpt = {'completed': [], 'results': {}}

# ── Pre-train Task A (if not already done) ──
TASK_A_MODEL_PATH = os.path.join(RESULTS_DIR, 'ewc_ablation_task_a.zip')

if task_a_score is None:
    print("\n[Phase 1] Pre-training Task A (standard Ant-v5, 300K steps)...")
    t0 = time.time()
    env_a = DummyVecEnv([make_env(1.0, 1.0)])
    model_a = PPO('MlpPolicy', env_a, verbose=0, device='cpu')
    model_a.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
    task_a_score = evaluate(model_a, make_env(1.0, 1.0))
    print(f"  Task A score: {task_a_score:.1f}  (took {(time.time()-t0)/60:.0f} min)")

    # Save
    model_a.save(TASK_A_MODEL_PATH)
    ckpt['task_a_score'] = task_a_score
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f)
else:
    print(f"\n[Phase 1] Loading pre-trained Task A (score: {task_a_score:.1f})...")
    model_a = PPO.load(TASK_A_MODEL_PATH)

# ── Estimate Fisher on Task A ──
print("\n[Phase 2] Estimating Fisher information on Task A...")
env_fisher = gym.make('Ant-v5')
obs, _ = env_fisher.reset()
fisher_dict = {}
for _ in range(N_REPLAY_SAMPLES):
    model_a.policy.zero_grad()
    obs_t = torch.FloatTensor(obs).unsqueeze(0)
    features = model_a.policy.features_extractor(obs_t)
    latent = model_a.policy.mlp_extractor.forward_actor(features)
    actions = model_a.policy.action_net(latent)
    (-actions.pow(2).sum()).backward(retain_graph=True)
    for name, param in model_a.policy.named_parameters():
        if param.grad is not None:
            fisher_dict[name] = fisher_dict.get(name, torch.zeros_like(param.data)) + param.grad.data ** 2
    obs, _, terminated, truncated, _ = env_fisher.step(env_fisher.action_space.sample())
    if terminated or truncated:
        obs, _ = env_fisher.reset()
env_fisher.close()
print(f"  Fisher estimated on {N_REPLAY_SAMPLES} samples")

# Save old params
old_params = {}
for name, param in model_a.policy.named_parameters():
    old_params[name] = param.data.clone()

# ── Run ablation for each λ ──
results = ckpt.get('results', {})

for lam in LAMBDA_VALUES:
    lam_str = str(lam)
    if lam_str in completed_lambdas:
        print(f"\n[λ={lam}] SKIP (already completed)")
        continue

    print(f"\n[λ={lam}] Starting...")
    t0 = time.time()

    # Load fresh copy of Task A model
    model = PPO.load(TASK_A_MODEL_PATH)

    # Task B: high gravity
    print(f"  Task B (high gravity, 1.5×)...")
    model = train_with_ewc_simple(
        model, make_env(1.5, 1.0), TRAIN_STEPS,
        old_params, fisher_dict, lam
    )

    # Update old_params and fisher for Task B→C transition
    # (Use current params as old_params for next task)
    old_params_b = {}
    for name, param in model.policy.named_parameters():
        old_params_b[name] = param.data.clone()

    # Estimate Fisher on Task B
    fisher_b = {}
    env_fb = gym.make('Ant-v5')
    env_fb.unwrapped.model.opt.gravity[2] = -14.715
    obs, _ = env_fb.reset()
    for _ in range(N_REPLAY_SAMPLES):
        model.policy.zero_grad()
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        features = model.policy.features_extractor(obs_t)
        latent = model.policy.mlp_extractor.forward_actor(features)
        actions = model.policy.action_net(latent)
        (-actions.pow(2).sum()).backward(retain_graph=True)
        for name, param in model.policy.named_parameters():
            if param.grad is not None:
                fisher_b[name] = fisher_b.get(name, torch.zeros_like(param.data)) + param.grad.data ** 2
        obs, _, terminated, truncated, _ = env_fb.step(env_fb.action_space.sample())
        if terminated or truncated:
            obs, _ = env_fb.reset()
    env_fb.close()

    # Task C: low friction
    print(f"  Task C (low friction, 0.3×)...")
    model = train_with_ewc_simple(
        model, make_env(1.0, 0.3), TRAIN_STEPS,
        old_params_b, fisher_b, lam
    )

    # Evaluate retention on Task A
    retention_score = evaluate(model, make_env(1.0, 1.0))
    retention_pct = (retention_score / task_a_score) * 100

    elapsed = (time.time() - t0) / 60
    results[lam_str] = {
        'retention_score': retention_score,
        'retention_pct': retention_pct,
        'time_min': elapsed,
    }
    completed_lambdas.add(lam_str)
    ckpt['completed'] = list(completed_lambdas)
    ckpt['results'] = results

    print(f"  → Retention: {retention_pct:.1f}% ({retention_score:.1f}/{task_a_score:.1f})")
    print(f"  → Time: {elapsed:.0f} min")

    # Save checkpoint after each λ
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)

    # Save intermediate results JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'task_a_score': task_a_score, 'results': results}, f, indent=2)

# ── Naive baseline (no EWC) ──
if 'naive' not in completed_lambdas:
    print("\n[Naive] No EWC baseline...")
    t0 = time.time()
    model = PPO.load(TASK_A_MODEL_PATH)
    model = train_with_ewc_simple(
        model, make_env(1.5, 1.0), TRAIN_STEPS,
        old_params, fisher_dict, lam=0  # lam=0 disables EWC
    )
    model = train_with_ewc_simple(
        model, make_env(1.0, 0.3), TRAIN_STEPS,
        old_params, fisher_dict, lam=0
    )
    naive_score = evaluate(model, make_env(1.0, 1.0))
    naive_pct = (naive_score / task_a_score) * 100
    elapsed = (time.time() - t0) / 60
    ckpt['naive'] = {'retention_score': naive_score, 'retention_pct': naive_pct}

    print(f"  Naive retention: {naive_pct:.1f}% (took {elapsed:.0f} min)")

    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(ckpt, f, indent=2)

# ── Plot ──
print("\n[Plot] Generating ablation plot...")

# Filter completed results
lambdas_done = []
retentions = []
for lam in LAMBDA_VALUES:
    lam_str = str(lam)
    if lam_str in results:
        lambdas_done.append(lam)
        retentions.append(results[lam_str]['retention_pct'])

if lambdas_done:
    fig, ax = plt.subplots(figsize=(10, 5))

    # EWC bars
    colors = plt.cm.Blues(0.3 + 0.6 * np.array(retentions) / max(retentions))
    ax.bar(range(len(lambdas_done)), retentions, color=colors, edgecolor='white', width=0.6)

    # Naive baseline
    if 'naive' in ckpt:
        ax.axhline(y=ckpt['naive']['retention_pct'], color='#F44336', ls='--', lw=1.5,
                   label=f"Naive ({ckpt['naive']['retention_pct']:.1f}%)")

    # Task A baseline
    ax.axhline(y=100, color='#4CAF50', ls='--', lw=1, alpha=0.5, label='Task A baseline (100%)')

    # 80% threshold
    ax.axhline(y=80, color='gray', ls=':', lw=1, alpha=0.4, label='80% threshold')

    ax.set_xticks(range(len(lambdas_done)))
    ax.set_xticklabels([f'λ={lam}' for lam in lambdas_done])
    ax.set_ylabel('Task A Retention (%)')
    ax.set_title(f'EWC λ Ablation on Ant-v5 ({TRAIN_STEPS//1000}K steps × 3 tasks)')
    ax.legend(loc='lower right')

    for i, (lam, ret) in enumerate(zip(lambdas_done, retentions)):
        ax.text(i, ret + 1, f'{ret:.1f}%', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ewc_ablation.png'), dpi=150)
    print(f"  Saved: ewc_ablation.png")

print("\n" + "="*60)
print("DONE!")
print(f"Results saved to: {RESULTS_FILE}")
print(f"Plot saved to: ewc_ablation.png")
print("="*60)
