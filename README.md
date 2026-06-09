# Twins Architecture

**Neurodevelopment as Blueprint: Biological Validation of the Twins Architecture**

> A developmental approach to AGI — frozen reflex layer + independent learning layer + token migration.  
> All experiments run on a standard laptop. No GPU required.

---

## What is this?

The human brain doesn't wake up intelligent. It follows a precise developmental sequence:

1. **Birth**: only brainstem reflexes (factory pre-installed)
2. **Ages 0-3**: synaptic explosion, learning mechanisms come online
3. **Ages 3+**: synaptic pruning, verified circuits are frozen
4. **Ages 3-25**: prefrontal cortex slowly myelinates (social cognition built on physical cognition)

The Twins Architecture applies this same logic to AGI:

```
┌─ Reflex Layer ─┐     ┌─ Learning Layer ─┐
│  Frozen, fast   │ ←→ │  Trainable, slow  │
│  Pre-trained    │     │  From-scratch     │
│  Immediate       │     │  Adaptive         │
└────────────────┘     └──────────────────┘
         ↑                      │
         └── Token Migration ───┘
         (verified knowledge flows upward)
```

**Key insight discovered during experiments**: fine-tuning the reflex layer on novel data causes *negative transfer* — the frozen reflex, unchanged, outperforms any fine-tuning. The optimal strategy is: keep reflex permanently frozen, train the learning layer independently from scratch, and route between them by confidence.

---

## Quick Start

### Requirements

```bash
pip install gymnasium stable-baselines3 torch numpy matplotlib scikit-learn
```

A 2013 MacBook (Intel i5, no GPU) is sufficient for all experiments.

### Run the flagship experiment

```bash
cd experiments
python experiment_twins.py
```

This runs the complete Twins Architecture lifecycle in under 5 minutes:

```
Reflex(standard)      R² = 0.377   ← Pre-trained
Reflex(novel)         R² = 0.368   ← Degraded, activates learning
Learning(from-scratch)R² = 0.451   ← Surpasses reflex
Reflex(after migrate) R² = 0.451   ← Updated via Token Migration
Static baseline       R² = 0.373   ← Never improves, forever stuck
```

---

## Experiments

| # | Script | What It Tests | Time |
|---|--------|---------------|------|
| 5.1 | `experiment_baselines.py` | EWC/SI/MAS/Naive anti-forgetting on Ant-v5 (3 tasks, 300K steps each) | ~6h |
| — | `experiment_ewc_ablation.py` | EWC λ sweep (7 values, 300K steps each) | ~3h |
| — | `experiment_fidelity.py` | Training stability: EWC vs Naive over 300K steps | ~2h |
| 5.2 | `experiment_curiosity.py` | Curiosity-driven exploration (Ant-v5, 500K pretrain + 100K explore) | ~5h |
| 5.3 | `experiment_multimodal.py` | Cross-modal prediction: aligned vs time-shifted (5 seeds) | ~5min |
| 5.4 | `experiment_warmstart.py` | Warm-start vs frozen vs scratch on novel gravity (5 seeds) | ~3min |
| 5.5 | `experiment_novelty_gradient.py` | 5 gravity levels × 4 strategies — finds crossover threshold | ~15min |
| 5.6 | `experiment_twins.py` | 📍 **End-to-end closed loop** (complete architecture lifecycle) | ~5min |

All experiments output figures to the `results/` directory.

---

## Generate the Paper

```bash
python gen_paper.py
```

Produces both Chinese and English `.docx` manuscripts, with all experiment figures embedded.

---

## Key Findings

1. **EWC prevents catastrophic forgetting** — retention 93.2% vs Naive 71.7% (λ ablation confirms robustness across 3 orders of magnitude)
2. **Curiosity drives 6.7× deeper exploration** than extrinsic reward alone
3. **Multimodal alignment requires temporal synchrony** — V→F prediction drops from R²=0.513 to −0.509 when time-shifted by 20 steps
4. **Negative transfer is real and measurable** — the crossover threshold where scratch surpasses warm-start sits at ~1.25-1.50× domain discrepancy
5. **The Twins closed loop works** — a 5-minute experiment on consumer hardware demonstrates the full lifecycle: degrade → learn → migrate → improve

---

## Citation

```bibtex
@article{chennan2026twins,
  title={Neurodevelopment as Blueprint: Biological Validation of the Twins Architecture},
  author={Chennan},
  journal={Frontiers in Robotics and AI, section Computational Intelligence in Robotics},
  year={2026},
  note={Manuscript ID: 1902952, under review}
}
```

Preprint: [Zenodo 10.5281/zenodo.20572427](https://doi.org/10.5281/zenodo.20572427)

---

## Author

Independent researcher. All experiments designed and executed on a consumer MacBook (Intel Core i5, 1.6GHz, no GPU).

Contact: cn85608869@gmail.com
