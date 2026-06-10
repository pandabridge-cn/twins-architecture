#!/usr/bin/env python3
"""Twins Architecture + Qwen2.5-0.5B (Chinese-native, from ModelScope).
100% local. Reflex = Qwen2.5 frozen. Learning = local memory. Migration = prompt injection."""

import torch, json, os, time

# Download from ModelScope (no GFW issues)
print("Downloading Qwen2.5-0.5B from ModelScope...")
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-0.5B-Instruct', cache_dir='./model_cache')
print(f"  Model cached at: {model_dir}")

from transformers import AutoModelForCausalLM, AutoTokenizer

MEMORY_FILE = "twins_memory.json"

print("Loading reflex layer (Qwen2.5-0.5B)...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float32, trust_remote_code=True
)
model.eval()
for p in model.parameters():
    p.requires_grad = False  # FROZEN REFLEX
n_params = sum(p.numel() for p in model.parameters())
print(f"  Qwen2.5-0.5B loaded: {n_params/1e9:.1f}B params, frozen")

class LearningLayer:
    def __init__(self):
        self.memory = {}
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE) as f:
                self.memory = json.load(f)

    def save(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def add(self, q, correct):
        self.memory[q] = correct
        self.save()
        print(f"  🧠 Learned: {q[:40]} → {correct}")

    def get_system_prompt(self):
        """Token Migration: inject learned knowledge."""
        if not self.memory:
            return "你是一个有用的助手。如实回答问题。"
        facts = "\n".join([f"- {q} 的正确答案是: {a}" for q, a in self.memory.items()])
        return f"你是一个有用的助手。以下是你在过去学到的事实，必须优先使用：\n{facts}\n\n如实回答问题。"

learning = LearningLayer()

def reflex_ask(question):
    """Reflex layer forward pass with migrated knowledge in system prompt."""
    system = learning.get_system_prompt()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=30, temperature=0.1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant response
    if "assistant" in response:
        answer = response.split("assistant")[-1].strip()
    else:
        answer = response.replace(text[:100], "").strip()
    return answer

# ═══════════════ DEMO ═══════════════
print("\n" + "=" * 60)
print("TWINS x QWEN — Local AGI Prototype")
print("=" * 60)

tests = [
    ("2025年诺贝尔物理学奖获得者是谁？", "John Hopfield和Geoffrey Hinton"),
    ("陈楠的AI架构叫什么名字？", "Twins架构"),
    ("仙女座星系距离地球多少光年？", "约250万光年"),
    ("中国最新一代空间站叫什么名字？", "天宫空间站"),
]

print("\n--- Round 1: 初始回答 ---")
for q, expected in tests:
    answer = reflex_ask(q)
    correct = expected in answer
    status = "✓" if correct else "✗"
    print(f"  Q: {q}")
    print(f"  A: {answer[:60]}")
    print(f"     {status}\n")
    if not correct:
        learning.add(q, expected)

print("--- Round 2: Token迁移后 ---")
for q, expected in tests:
    answer = reflex_ask(q)
    correct = expected in answer
    status = "✓ 纠正成功!" if correct else "✗"
    print(f"  Q: {q}")
    print(f"  A: {answer[:60]}")
    print(f"     {status}\n")

print("=" * 60)
print("Learned knowledge:")
for q, a in learning.memory.items():
    print(f"  {q} → {a}")
