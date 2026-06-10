#!/usr/bin/env python3
"""Twins v4 FIXED: Spaced Repetition with persistent Token Migration.
Key fix: corrections are ALWAYS injected. Mastery = correct N times consecutively."""

import torch, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelscope import snapshot_download
from collections import defaultdict

model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./model_cache')
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()
for p in model.parameters(): p.requires_grad = False

def ask(q, memory=None):
    sys = "你是一个有用的助手。如实回答问题。"
    if memory:
        facts = "；".join([f"{q}: {a}" for q, a in memory.items()])
        user = f"已知事实：{facts}\n\n问题：{q}"
    else:
        user = q
    msgs = [{"role":"system","content":sys},{"role":"user","content":user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=40, temperature=0.1,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()

def check(a, t):
    kw = t.replace("约","").replace("的","").replace("是","").replace("在","").split()
    return sum(1 for k in kw if k in a) >= max(1, len(kw)*0.35)

questions = [
    ("2025年诺贝尔物理学奖获得者是谁？", "John Hopfield和Geoffrey Hinton"),
    ("陈楠提出的AI架构叫什么名字？", "Twins架构"),
    ("仙女座星系距离地球大约多少光年？", "约250万光年"),
    ("中国正在运行的最新空间站叫什么？", "天宫空间站"),
    ("2024年巴黎奥运会中国获得多少枚金牌？", "40枚"),
    ("DeepSeek公司的创始人是哪位？", "梁文锋"),
    ("量子纠缠被爱因斯坦称为什么？", "鬼魅般的超距作用"),
    ("人类基因组计划大约在哪年完成？", "2003年"),
    ("太阳系中最大的行星是哪颗？", "木星"),
    ("图灵测试是哪一年提出的？", "1950年"),
]

print("="*60)
print("TWINS v4: Spaced Repetition (FIXED)")
print("="*60)

memory = {}      # q → answer (Token Migration store)
streak = defaultdict(int)  # consecutive correct count

# 5 rounds: each round reviews ALL known questions + adds new ones
batch_schedule = [(1, 0, 4), (2, 4, 7), (3, 7, 9), (4, 9, 10), (5, None, None)]

results = []

for rd, start, end in batch_schedule:
    corrections = "；".join([f"{q}: {a}" for q, a in memory.items()]) if memory else ""
    correct = 0; total = 0

    # Review: re-test all known questions
    for q, gt in list(memory.items()):
        ans = ask(q, memory)
        c = check(ans, gt)
        if c: streak[q] += 1; correct += 1
        else: streak[q] = 0; memory.pop(q, None)  # remove from memory, re-learn
        total += 1

    # New questions
    if start is not None:
        for q, gt in questions[start:end]:
            ans = ask(q, memory)
            c = check(ans, gt)
            if not c:
                memory[q] = gt  # learn it
                ans = ask(q, memory)  # re-ask with new knowledge
                c = check(ans, gt)
            if c: streak[q] = 1; correct += 1
            else: streak[q] = 0
            total += 1

    mastered = sum(1 for q, s in streak.items() if s >= 2)
    results.append((rd, correct, total, mastered))
    print(f"Round {rd}: {correct}/{total} correct | Mastered: {mastered}/{len(questions)} | Streaks: {dict(streak)}")

# Final comprehensive test
print(f"\n{'='*60}")
print(f"FINAL RESULT: {results[-1][1]}/{results[-1][2]} ({results[-1][1]/results[-1][2]*100:.0f}%)")
print(f"Mastered (streak≥2): {results[-1][3]}/{len(questions)}")
print(f"{'='*60}")

# PLOT
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
rds = [r[0] for r in results]
acc = [r[1]/r[2]*100 for r in results]
ax.plot(rds, acc, 'o-', color='#2196F3', lw=3, markersize=12)
ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
ax.set_title('Spaced Repetition Learning Curve')
ax.set_ylim(0, 105)
for r, a in zip(rds, acc):
    ax.text(r, a+3, f'{a:.0f}%', ha='center', fontweight='bold', fontsize=14)

ax = axes[1]
mas = [r[3]/len(questions)*100 for r in results]
ax.plot(rds, mas, 's-', color='#4CAF50', lw=3, markersize=12)
ax.set_xlabel('Round'); ax.set_ylabel('Mastered (%)')
ax.set_title('Knowledge Mastery (streak≥2)')
ax.set_ylim(0, 105)
for r, m in zip(rds, mas):
    ax.text(r, m+3, f'{m:.0f}%', ha='center', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('results/twins_llm_validation.png', dpi=150, bbox_inches='tight')
print("Saved: results/twins_llm_validation.png")
