"""
Experiment 5 — AdaptEvolve: Confidence-Based Model Routing
===========================================================
Implements adaptive LLM selection inspired by AdaptEvolve paper.
Routes queries between small and large models based on token-level
confidence scores (logprobs) — the same approach used in the paper.

Two conditions compared across 100 industrial tasks:

Condition 1 — BASELINE (Fixed large model):
  Always use 397B-cloud regardless of task complexity.
  Maximum quality, maximum cost.

Condition 2 — AdaptEvolve (Confidence cascade):
  Start with smallest model (TinyLlama 1.1B).
  Calculate mean logprob confidence from generated tokens.
  If confidence >= threshold → accept answer, stop cascade.
  If confidence < threshold → escalate to next model.
  Cascade: TinyLlama → Phi → Mistral → Qwen3.5 0.8B → 2B → 4B → 9B → 397B

KPIs:
  - Step coverage quality  : was the final answer good?
  - Average latency        : how fast was the system?
  - Escalation rate        : how often was a larger model needed?
  - Model usage distribution: which models handled which tasks?
"""

import requests
import time
import json
import os
import re
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_FILE = os.path.join(SCRIPT_DIR, "tasks.json")
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots_exp5")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}\n")

# Single confidence threshold — pure confidence-based routing
# No hardcoding per category — the confidence signal alone drives routing
# 0.63 is the sweet spot based on observed logprob ranges:
#   TinyLlama on simple tasks:  0.85-1.00 → accepts immediately
#   TinyLlama on complex tasks: 0.55-0.69 → escalates to Phi
#   Phi on complex tasks:       0.60-0.70 → escalates to Mistral
#   Mistral on complex tasks:   0.70-0.80 → accepts at threshold
# Expected escalation rate: 35-50%
CONFIDENCE_THRESHOLD = 0.63

# Cascade order — small to large (all local, no cloud dependency)
CASCADE_MODELS = [
    {"name": "tinyllama:1.1b",  "label": "TinyLlama\n1.1B",  "params_b": 1.1},
    {"name": "phi:latest",      "label": "Phi\n2.7B",         "params_b": 2.7},
    {"name": "mistral:latest",  "label": "Mistral\n7B",       "params_b": 7.0},
    {"name": "qwen3.5:0.8b",    "label": "Qwen3.5\n0.8B",    "params_b": 0.8},
    {"name": "qwen3.5:2b",      "label": "Qwen3.5\n2B",      "params_b": 2.0},
    {"name": "qwen3.5:4b",      "label": "Qwen3.5\n4B",      "params_b": 4.0},
    {"name": "qwen3.5:9b",      "label": "Qwen3.5\n9B",      "params_b": 9.0},
]

BASELINE_MODEL = "qwen3.5:9b"  # largest local model — baseline always uses this

# ============================================================
# LOAD TASKS
# ============================================================

with open(TASKS_FILE) as f:
    ALL_TASKS = json.load(f)

print(f"Loaded {len(ALL_TASKS)} tasks from {TASKS_FILE}")
print(f"  Simple:  {sum(1 for t in ALL_TASKS if t['category'] == 'simple')}")
print(f"  Medium:  {sum(1 for t in ALL_TASKS if t['category'] == 'medium')}")
print(f"  Complex: {sum(1 for t in ALL_TASKS if t['category'] == 'complex')}")
print(f"  Coding:  {sum(1 for t in ALL_TASKS if t['category'] == 'coding')}")
print()

# ============================================================
# GENERATION WITH LOGPROBS
# Uses /api/generate endpoint which supports logprobs
# ============================================================

def get_timeout(model_name):
    if "cloud"    in model_name: return 120
    elif "9b"     in model_name: return 180
    elif "4b"     in model_name: return 120
    elif "mistral" in model_name: return 120
    else:                        return 90

def is_qwen(model_name):
    return "qwen" in model_name.lower()

OLLAMA_GENERATE = "http://localhost:11434/api/generate"

def generate_with_confidence(model_name, prompt, max_tokens=250):
    """
    Generate a response and calculate genuine confidence score from logprobs.
    Uses /api/generate with logprobs=True for local models.
    Confidence = mean(exp(logprob)) across all generated tokens.
    Higher confidence = model was consistently sure about its tokens.
    Lower confidence = model was uncertain, should escalate.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens
        }
    }
    # think=False must be at TOP LEVEL of payload for qwen models
    if is_qwen(model_name):
        payload["think"] = False

    timeout = get_timeout(model_name)

    for attempt in range(2):
        start = time.time()
        try:
            r = requests.post(OLLAMA_GENERATE, json=payload, timeout=timeout)
            r.raise_for_status()
            elapsed = round(time.time() - start, 2)
            data = r.json()

            text = data.get("response", "").strip()
            # Strip think blocks from qwen models
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

            if not text:
                print(f"  [DEBUG] Keys: {list(data.keys())}")
                print(f"  [DEBUG] response: '{data.get('response','MISSING')[:100]}'")
                print(f"  [WARN] Empty response attempt {attempt+1}, retrying...")
                time.sleep(3)
                continue

            # Calculate genuine confidence from logprobs
            # logprob is log(probability) — convert back to probability with exp()
            # Mean probability across all tokens = overall confidence
            confidence = 0.5  # default if logprobs unavailable
            logprobs_data = data.get("logprobs", [])
            if logprobs_data and isinstance(logprobs_data, list):
                lp_values = []
                for token_info in logprobs_data:
                    if isinstance(token_info, dict):
                        lp = token_info.get("logprob", None)
                        if lp is not None:
                            lp_values.append(lp)
                if lp_values:
                    mean_lp   = sum(lp_values) / len(lp_values)
                    confidence = round(math.exp(mean_lp), 4)

            return text, confidence, elapsed, len(text.split())

        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt == 0:
                time.sleep(5)

    return "", 0.0, 0.0, 0

# ============================================================
# PROMPT
# ============================================================

def build_prompt(task):
    return (
        "You are an industrial engineering expert.\n"
        "Answer the following task or question clearly and concisely.\n"
        "Provide a structured response with numbered steps if applicable.\n\n"
        f"TASK: {task}\n\nANSWER:"
    )

# ============================================================
# SIMPLE QUALITY SCORING
# Since tasks are general industrial (not battery specific),
# we use response length and structure as quality proxy.
# A good response has at least 30 words and some structure.
# ============================================================

def score_response(text, task_category):
    """
    Score response quality based on category expectations.
    Returns 0.0 to 1.0
    """
    if not text:
        return 0.0

    word_count = len(text.split())
    has_numbers = bool(re.search(r'\d+[\.\)]', text))
    has_keywords = any(kw in text.lower() for kw in [
        "system", "process", "control", "data", "sensor", "machine",
        "safety", "monitor", "implement", "design", "analyze", "operate",
        "logic", "algorithm", "network", "automation", "quality"
    ])

    if task_category == "simple":
        # Simple tasks: just needs a decent answer, 20+ words
        if word_count >= 30: return 1.0
        if word_count >= 20: return 0.8
        if word_count >= 10: return 0.6
        return 0.4

    elif task_category == "medium":
        # Medium: needs explanation, 50+ words
        if word_count >= 80 and has_keywords: return 1.0
        if word_count >= 50: return 0.8
        if word_count >= 30: return 0.6
        return 0.4

    elif task_category == "complex":
        # Complex: needs steps and detail, 100+ words
        if word_count >= 120 and has_numbers and has_keywords: return 1.0
        if word_count >= 80 and has_numbers: return 0.8
        if word_count >= 50: return 0.6
        return 0.4

    elif task_category == "coding":
        # Coding: needs logic explanation, 80+ words
        if word_count >= 100 and has_numbers and has_keywords: return 1.0
        if word_count >= 70 and has_keywords: return 0.8
        if word_count >= 40: return 0.6
        return 0.4

    return 0.6

def avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else 0.0

# ============================================================
# CONDITION 1 — BASELINE: Always use 397B-cloud
# ============================================================

print("=" * 60)
print(f"CONDITION 1 — BASELINE (Always {BASELINE_MODEL})")
print("=" * 60)

baseline_results = []

for i, task_info in enumerate(ALL_TASKS):
    task     = task_info["task"]
    category = task_info["category"]

    print(f"\n  [{i+1:3d}/100] [{category:7s}]")
    print(f"  TASK: {task}")
    print(f"  ─────────────────────────────────────────────")

    text, conf, lat, tok = generate_with_confidence(
        BASELINE_MODEL, build_prompt(task)
    )
    quality = score_response(text, category)

    # Show the generated answer
    print(f"  ANSWER:\n{text}")
    print(f"  → quality={quality}  conf={conf}  lat={lat}s  tok={tok}")

    baseline_results.append({
        "task":     task,
        "category": category,
        "model":    BASELINE_MODEL,
        "quality":  quality,
        "confidence": conf,
        "latency":  lat,
        "tokens":   tok
    })

print(f"\n  ── BASELINE SUMMARY ──")
print(f"  Avg quality:  {avg([r['quality']  for r in baseline_results])}")
print(f"  Avg latency:  {avg([r['latency']  for r in baseline_results])}s")
print(f"  Avg tokens:   {avg([r['tokens']   for r in baseline_results])}")

# ============================================================
# CONDITION 2 — AdaptEvolve cascade
# ============================================================

print("\n" + "=" * 60)
print(f"CONDITION 2 — AdaptEvolve (Single threshold={CONFIDENCE_THRESHOLD})")
print(f"  Pure confidence-based routing — no category hardcoding")
print("=" * 60)

adaptevolve_results = []

for i, task_info in enumerate(ALL_TASKS):
    task     = task_info["task"]
    category = task_info["category"]

    print(f"\n  [{i+1:3d}/100] [{category:7s}] threshold={CONFIDENCE_THRESHOLD}")
    print(f"  TASK: {task}")
    print(f"  ─────────────────────────────────────────────")

    final_text   = ""
    final_model  = ""
    final_conf   = 0.0
    final_lat    = 0.0
    final_tok    = 0
    escalations  = 0
    models_tried = []

    for model_info in CASCADE_MODELS:
        model_name = model_info["name"]
        models_tried.append(model_name)

        text, conf, lat, tok = generate_with_confidence(
            model_name, build_prompt(task)
        )

        print(f"    → {model_name:25s} conf={conf:.4f}  lat={lat}s", end="")

        final_text  = text
        final_model = model_name
        final_conf  = conf
        final_lat  += lat
        final_tok   = tok

        if conf >= CONFIDENCE_THRESHOLD:
            print(f"  ✅ ACCEPTED")
            print(f"  ANSWER:\n{text}")
            break
        else:
            print(f"  ↑ escalate")
            escalations += 1
            if model_name != CASCADE_MODELS[-1]["name"]:
                time.sleep(1)

    quality = score_response(final_text, category)

    adaptevolve_results.append({
        "task":        task,
        "category":    category,
        "final_model": final_model,
        "quality":     quality,
        "confidence":  final_conf,
        "latency":     final_lat,
        "tokens":      final_tok,
        "escalations": escalations,
        "models_tried": models_tried
    })

    print(f"    ── Final: {final_model}  quality={quality}  total_lat={round(final_lat,2)}s  escalations={escalations}")

# Summary
print(f"\n  ── AdaptEvolve SUMMARY ──")
print(f"  Avg quality:       {avg([r['quality']    for r in adaptevolve_results])}")
print(f"  Avg latency:       {avg([r['latency']    for r in adaptevolve_results])}s")
print(f"  Avg escalations:   {avg([r['escalations'] for r in adaptevolve_results])}")

model_usage = Counter(r["final_model"] for r in adaptevolve_results)
print(f"  Model usage:")
for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
    pct = round(count / len(adaptevolve_results) * 100)
    print(f"    {model:30s}: {count:3d} tasks ({pct}%)")

escalation_rate = sum(1 for r in adaptevolve_results if r["escalations"] > 0)
print(f"  Escalation rate: {escalation_rate}/{len(adaptevolve_results)} tasks ({round(escalation_rate/len(adaptevolve_results)*100)}%)")

# ============================================================
# PLOTTING
# ============================================================

categories   = ["simple", "medium", "complex", "coding"]
cat_labels   = ["Simple", "Medium", "Complex", "Coding"]
C_BASE       = "#888780"
C_ADAPT      = "#185FA5"

# ── Plot 1: Quality comparison overall + per category ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall
ax = axes[0]
base_qual  = avg([r["quality"] for r in baseline_results])
adapt_qual = avg([r["quality"] for r in adaptevolve_results])
bars = ax.bar(["Baseline\n(397B always)", "AdaptEvolve\n(cascade)"],
              [base_qual, adapt_qual],
              color=[C_BASE, C_ADAPT], alpha=0.85, width=0.4)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Average Quality Score (0-1)", fontsize=11)
ax.set_title("Overall Quality Comparison", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Per category
ax = axes[1]
x = np.arange(len(categories))
w = 0.35
base_by_cat  = [avg([r["quality"] for r in baseline_results    if r["category"] == c]) for c in categories]
adapt_by_cat = [avg([r["quality"] for r in adaptevolve_results if r["category"] == c]) for c in categories]

ax.bar(x - w/2, base_by_cat,  w, color=C_BASE,  alpha=0.85, label="Baseline")
ax.bar(x + w/2, adapt_by_cat, w, color=C_ADAPT, alpha=0.90, label="AdaptEvolve")
for i, (v1, v2) in enumerate(zip(base_by_cat, adapt_by_cat)):
    delta = v2 - v1
    col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
    ax.text(x[i], max(v1, v2) + 0.05, f"{delta:+.2f}",
            ha="center", fontsize=9, color=col, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cat_labels, fontsize=10)
ax.set_ylim(0, 1.2)
ax.set_ylabel("Quality Score", fontsize=11)
ax.set_title("Quality by Task Category", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.suptitle("Experiment 5 — AdaptEvolve: Quality Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "exp5_quality.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved: {path1}")

# ── Plot 2: Latency comparison ──
fig, ax = plt.subplots(figsize=(10, 5))
base_lat  = avg([r["latency"] for r in baseline_results])
adapt_lat = avg([r["latency"] for r in adaptevolve_results])
bars = ax.bar(["Baseline\n(397B always)", "AdaptEvolve\n(cascade)"],
              [base_lat, adapt_lat],
              color=[C_BASE, C_ADAPT], alpha=0.85, width=0.4)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{bar.get_height():.1f}s", ha="center", fontsize=12, fontweight="bold")
saving_pct = round((base_lat - adapt_lat) / base_lat * 100) if base_lat > 0 else 0
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title(f"Latency Comparison — AdaptEvolve saves ~{saving_pct}% latency",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "exp5_latency.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path2}")

# ── Plot 3: Model usage distribution (AdaptEvolve) ──
fig, ax = plt.subplots(figsize=(12, 5))
model_names_ordered = [m["name"]  for m in CASCADE_MODELS]
model_labels_ordered = [m["label"] for m in CASCADE_MODELS]
usage_counts = [model_usage.get(m, 0) for m in model_names_ordered]
colors = ["#B4B2A9" if "cloud" not in m else "#185FA5" for m in model_names_ordered]
bars = ax.bar(range(len(model_names_ordered)), usage_counts, color=colors, alpha=0.85)
for bar, count in zip(bars, usage_counts):
    if count > 0:
        pct = round(count / len(adaptevolve_results) * 100)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{count}\n({pct}%)", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(model_names_ordered)))
ax.set_xticklabels(model_labels_ordered, fontsize=9)
ax.set_ylabel("Number of Tasks Handled", fontsize=11)
ax.set_title("AdaptEvolve — Model Usage Distribution\n(which model gave the final answer)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "exp5_model_usage.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path3}")

# ── Plot 4: Escalation rate by category ──
fig, ax = plt.subplots(figsize=(10, 5))
esc_by_cat = []
for cat in categories:
    cat_results = [r for r in adaptevolve_results if r["category"] == cat]
    esc_count = sum(1 for r in cat_results if r["escalations"] > 0)
    esc_by_cat.append(round(esc_count / len(cat_results) * 100) if cat_results else 0)

bars = ax.bar(cat_labels, esc_by_cat, color=C_ADAPT, alpha=0.85)
for bar, val in zip(bars, esc_by_cat):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Escalation Rate (%)", fontsize=11)
ax.set_ylim(0, 110)
ax.set_title("Escalation Rate by Task Category\n(% of tasks that needed a larger model)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path4 = os.path.join(PLOTS_DIR, "exp5_escalation_rate.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path4}")

# ── Save JSON results ──
results_data = {
    "config": {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "routing": "pure_confidence_based_no_hardcoding",
        "total_tasks": len(ALL_TASKS),
        "cascade_models": [m["name"] for m in CASCADE_MODELS]
    },
    "baseline": {
        "avg_quality":  avg([r["quality"]  for r in baseline_results]),
        "avg_latency":  avg([r["latency"]  for r in baseline_results]),
        "avg_tokens":   avg([r["tokens"]   for r in baseline_results]),
    },
    "adaptevolve": {
        "avg_quality":    avg([r["quality"]    for r in adaptevolve_results]),
        "avg_latency":    avg([r["latency"]    for r in adaptevolve_results]),
        "avg_tokens":     avg([r["tokens"]     for r in adaptevolve_results]),
        "avg_escalations": avg([r["escalations"] for r in adaptevolve_results]),
        "escalation_rate": round(escalation_rate / len(adaptevolve_results) * 100, 1),
        "model_usage":    dict(model_usage),
    },
    "per_task": {
        "baseline":     baseline_results,
        "adaptevolve":  adaptevolve_results
    }
}

path_json = os.path.join(PLOTS_DIR, "exp5_results.json")
with open(path_json, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"✅ Saved: {path_json}")

print("\n" + "=" * 60)
print("EXPERIMENT 5 COMPLETE")
print("=" * 60)
print(f"\nKEY RESULTS:")
print(f"  Baseline  quality: {avg([r['quality'] for r in baseline_results])}")
print(f"  AdaptEvolve quality: {avg([r['quality'] for r in adaptevolve_results])}")
print(f"  Baseline  latency: {avg([r['latency'] for r in baseline_results])}s")
print(f"  AdaptEvolve latency: {avg([r['latency'] for r in adaptevolve_results])}s")
print(f"  Escalation rate: {round(escalation_rate/len(adaptevolve_results)*100)}%")
print(f"  Latency saving: ~{saving_pct}%")