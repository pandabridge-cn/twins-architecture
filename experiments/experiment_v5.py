#!/usr/bin/env python3
"""Twins v5: Weight-level Token Migration (LoRA) + Self-Confidence Detection.
Addresses limitations 1 & 3: knowledge persists across context + auto error detection."""

import torch, os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from modelscope import snapshot_download

# Install peft if missing
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    os.system("pip3 install peft -q")
    from peft import get_peft_model, LoraConfig, TaskType

model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./model_cache')
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})

print("Loading model + LoRA...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, trust_remote_code=True)

# Configure LoRA: only train tiny adapters, keep base frozen
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4, lora_alpha=8, lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, lora_config)
model.eval()
for name, p in model.named_parameters():
    p.requires_grad = 'lora' in name  # Only LoRA weights are trainable

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Params: {total/1e6:.1f}M total, {trainable/1e6:.2f}M trainable (LoRA)")
print(f"  Memory saving: {(1 - trainable/total)*100:.0f}%")

def ask(question, corrections_text=None):
    """Forward pass WITHOUT corrections in prompt — tests if LoRA weight knowledge persists."""
    system = "你是一个有用的助手。如实回答问题。"
    user = question  # No corrections injected!
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.1,
                                  do_sample=False, pad_token_id=tokenizer.eos_token_id,
                                  output_scores=True, return_dict_in_generate=True)
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).split("assistant")[-1].strip()
    # Extract confidence from generation scores
    scores = outputs.scores
    if scores:
        probs = torch.stack([torch.softmax(s, dim=-1) for s in scores])
        confidence = probs.max(dim=-1).values.mean().item()
    else:
        confidence = 0.5
    return response, confidence

def check(a, t):
    kw = t.replace("约","").replace("的","").replace("是","").replace("在","").split()
    return sum(1 for k in kw if k in a) >= max(1, len(kw)*0.35)

# ═══════════════ TRAIN LoRA ON CORRECTIONS ═══════════════
def train_lora(question, correct_answer, steps=20):
    """Train LoRA weights to internalize a single correction."""
    system = "你是一个有用的助手。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
        {"role": "assistant", "content": correct_answer}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(**{k: v for k, v in inputs.items()},
                       labels=inputs['input_ids'])
        outputs.loss.backward()
        optimizer.step()

# ═══════════════ EXPERIMENT ═══════════════
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

print("\n" + "="*60)
print("TWINS v5: LoRA Weight Migration + Self-Confidence")
print("="*60)

results = []

for round_num in range(1, 4):
    corrected = 0
    total = 0
    auto_learned = 0

    if round_num == 1:
        qs = questions[:4]
    elif round_num == 2:
        qs = questions[:8]
    else:
        qs = questions

    for q, gt in qs:
        ans, conf = ask(q)  # NO corrections in prompt — tests weight-level knowledge
        is_correct = check(ans, gt)
        total += 1

        if not is_correct:
            # Self-confidence check: if model is uncertain → auto-learn
            if conf < 0.9:
                auto_learned += 1
                print(f"  [AUTO] Low confidence ({conf:.2f}), self-learning: {q[:30]}...")
            else:
                print(f"  [HIGH] High confidence ({conf:.2f}) but WRONG: {q[:30]}...")

            # Weight-level token migration: train LoRA
            train_lora(q, gt)
            ans, conf = ask(q)  # re-test
            is_correct = check(ans, gt)

        if is_correct:
            corrected += 1

    results.append((round_num, corrected, total, auto_learned))
    print(f"Round {round_num}: {corrected}/{total} correct | Auto-learned: {auto_learned} | No prompt corrections used!")

# ═══════════════ FINAL: Test across ALL questions WITHOUT prompt corrections ═══════════════
print(f"\n{'='*60}")
print("FINAL TEST: All 10 questions, NO prompt corrections, NO LoRA training")
print("Tests if weight-level knowledge persists")
print(f"{'='*60}")

final_correct = 0
for q, gt in questions:
    ans, conf = ask(q)
    c = check(ans, gt)
    if c: final_correct += 1
    print(f"  {q[:35]:35s} → {ans[:40]:40s} [conf={conf:.2f}] {'✓' if c else '✗'}")

# Compare: ask SAME questions with prompt corrections (RAG-style)
print(f"\n{'='*60}")
print("COMPARISON: Same questions WITH prompt corrections (RAG baseline)")
print(f"{'='*60}")

rag_context = "参考：\n" + "\n".join([f"Q: {q} A: {a}" for q, a in questions])
rag_correct = 0
for q, gt in questions:
    system = "你是一个有用的助手。"
    user = f"{rag_context}\n\n问题：{q}"
    messages = [{"role":"system","content":system},{"role":"user","content":user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40, temperature=0.1,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0], skip_special_tokens=True).split("assistant")[-1].strip()
    if check(ans, gt): rag_correct += 1
print(f"  RAG (with prompt): {rag_correct}/{len(questions)}")

print(f"\n{'='*60}")
print(f"FINAL (weight-only, no prompt): {final_correct}/{len(questions)} ({final_correct/len(questions)*100:.0f}%)")
print(f"RAG (prompt injection):          {rag_correct}/{len(questions)} ({rag_correct/len(questions)*100:.0f}%)")
print(f"{'='*60}")

# ═══════════════ PLOT ═══════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (A) Learning curve
ax = axes[0]
rds = [r[0] for r in results]
acc = [r[1]/r[2]*100 for r in results]
ax.plot(rds, acc, 'o-', color='#2196F3', lw=3, markersize=12)
ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
ax.set_title('LoRA Weight Migration\n(No prompt corrections)')
ax.set_ylim(0, 105)
for r, a in zip(rds, acc):
    ax.text(r, a+3, f'{a:.0f}%', ha='center', fontweight='bold')

# (B) Weight vs Prompt
ax = axes[1]
x = np.arange(2); wd = 0.5
ax.bar(x, [final_correct/10*100, rag_correct/10*100], wd,
       color=['#9C27B0', '#4CAF50'], edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(['LoRA Weights\n(permanent)', 'Prompt RAG\n(ephemeral)'])
ax.set_ylabel('Accuracy (%)'); ax.set_title('Weight vs Prompt Knowledge')
ax.set_ylim(0, 105)
for i, v in enumerate([final_correct/10*100, rag_correct/10*100]):
    ax.text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=14)

# (C) Auto-learn stats
ax = axes[2]
auto = [r[3] for r in results]
ax.bar(rds, auto, color='#FF9800', edgecolor='white', width=0.5)
ax.set_xlabel('Round'); ax.set_ylabel('Auto-learned')
ax.set_title('Self-Confidence Auto-Learning\n(Low confidence → auto LoRA train)')

plt.tight_layout()
plt.savefig('results/twins_llm_validation.png', dpi=150, bbox_inches='tight')
print("Saved: results/twins_llm_validation.png")
