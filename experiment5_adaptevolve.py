"""
Experiment 5 — AdaptEvolve: Confidence-Based Model Routing
===========================================================
Implements adaptive LLM selection inspired by the AdaptEvolve paper.
Routes queries between small and large models based on token-level
confidence scores (logprobs) — pure confidence-based routing with
no hardcoded rules per task category.

Dataset: FailureSensorIQ (ibm-research/FailureSensorIQ)
  - Expert-curated MCQA benchmark for industrial assets
  - Built from ISO documents covering predictive maintenance,
    sensor fault detection, and failure mode reasoning
  - Single-true multiple choice questions (A/B/C/D)
  - Ground truth correct answers — quality = 1 (correct) or 0 (wrong)

Quality metric: ACCURACY (correct/incorrect vs ground truth)
  - FIXED: Previous proxy metric used word count which could not
    detect wrong answers. Now using objective ground truth.

Prompting: Chain of Thought (CoT)
  - Model asked to reason step by step before giving final answer
  - Final answer extracted as letter A/B/C/D from response

Two conditions compared:

Condition 1 — BASELINE (Fixed large model):
  Always use Qwen3.5 9B regardless of task complexity.

Condition 2 — AdaptEvolve (Confidence-based cascade):
  Start with smallest model (Qwen3.5 0.8B).
  Calculate mean logprob confidence from generated tokens.
  If confidence >= threshold (0.65) → accept answer → check ground truth
  If confidence < threshold → escalate to next larger model.
  Cascade strictly ascending: 0.8B → 1.1B → 2B → 2.7B → 4B → 7B → 9B

Calibration / Test split:
  Calibration set (first 100 questions) — used to select threshold 0.65
  Test set        (last  100 questions) — used to report final results

KPIs:
  - Accuracy          : % correct answers vs ground truth
  - Average latency   : total time including all escalation steps
  - Escalation rate   : % tasks needing more than one model
  - Model usage       : which models handled which tasks
  - Latency saving    : % improvement vs always using 9B
  - Overconfidence    : cases where confident but gave wrong answer
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

OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_GENERATE = "http://localhost:11434/api/generate"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots_exp5")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}\n")

# Confidence threshold selected on calibration set
CONFIDENCE_THRESHOLD = 0.65

# Cascade order — strictly ascending by parameter count
# 0.8B → 1.1B → 2.0B → 2.7B → 4.0B → 7.0B → 9.0B
CASCADE_MODELS = [
    {"name": "qwen3.5:0.8b",   "label": "Qwen3.5\n0.8B",  "params_b": 0.8},
    {"name": "tinyllama:1.1b", "label": "TinyLlama\n1.1B", "params_b": 1.1},
    {"name": "qwen3.5:2b",     "label": "Qwen3.5\n2B",     "params_b": 2.0},
    {"name": "phi:latest",     "label": "Phi\n2.7B",        "params_b": 2.7},
    {"name": "qwen3.5:4b",     "label": "Qwen3.5\n4B",     "params_b": 4.0},
    {"name": "mistral:latest", "label": "Mistral\n7B",      "params_b": 7.0},
    {"name": "qwen3.5:9b",     "label": "Qwen3.5\n9B",     "params_b": 9.0},
]

BASELINE_MODEL = "qwen3.5:9b"

# ============================================================
# LOAD FAILURESENSORIQ FROM HUGGING FACE
# ============================================================

def load_failuresensoriq(max_questions=200):
    """
    Load FailureSensorIQ single-true MCQA questions.
    Each question has: question text, 4 choices, 1 correct answer.
    Caches to disk to avoid re-downloading on re-runs.
    """
    cache_file = os.path.join(SCRIPT_DIR, "failuresensoriq_cache.json")

    if os.path.exists(cache_file):
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file) as f:
            questions = json.load(f)
        print(f"  Loaded {len(questions)} questions from cache")
        return questions[:max_questions]

    print("  Loading from Hugging Face (ibm-research/FailureSensorIQ)...")
    print("  Requires: pip install datasets")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "ibm-research/FailureSensorIQ",
            "single_true_multi_choice_qa",
            trust_remote_code=True
        )

        questions = []
        split = "test" if "test" in ds else "train"
        print(f"  Using split: {split} ({len(ds[split])} examples)")

        for item in ds[split]:
            q_text  = item.get("question", "")
            choices = item.get("choices", item.get("options", []))
            answer  = item.get("answer",  item.get("label", 0))

            if not q_text or not choices:
                continue

            if isinstance(answer, int):
                correct_letter = chr(65 + answer)
            else:
                correct_letter = str(answer).upper().strip()

            questions.append({
                "question":       q_text,
                "choices":        choices[:4],
                "correct_letter": correct_letter,
                "correct_index":  ord(correct_letter) - 65
            })

        print(f"  Loaded {len(questions)} questions")
        with open(cache_file, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"  Cached to: {cache_file}")
        return questions[:max_questions]

    except Exception as e:
        print(f"  [ERROR] Could not load dataset: {e}")
        print("  Run: pip install datasets")
        exit(1)


print("=" * 60)
print("Loading FailureSensorIQ benchmark dataset...")
print("=" * 60)
ALL_QUESTIONS = load_failuresensoriq(max_questions=200)
print(f"\nTotal questions loaded: {len(ALL_QUESTIONS)}")

# ============================================================
# CALIBRATION / TEST SPLIT
# First 100 used to select threshold (calibration)
# Last  100 used to report results (test)
# ============================================================

CALIBRATION_Q = ALL_QUESTIONS[:100]
TEST_Q        = ALL_QUESTIONS[100:]
EVAL_Q        = TEST_Q

print(f"\nCalibration : {len(CALIBRATION_Q)} questions (threshold selected here)")
print(f"Test set    : {len(EVAL_Q)} questions (results reported here)")
print(f"Threshold   : {CONFIDENCE_THRESHOLD} (selected on calibration set)")
print()

# ============================================================
# HELPERS
# ============================================================

def get_timeout(model_name):
    if "9b"       in model_name: return 180
    elif "4b"     in model_name: return 120
    elif "mistral" in model_name: return 120
    else:                         return 90

def is_qwen(model_name):
    return "qwen" in model_name.lower()

def avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0

# ============================================================
# CHAIN OF THOUGHT PROMPT
# ============================================================

def build_cot_prompt(question_data):
    """
    Build CoT prompt for multiple choice question.
    Asks model to reason step by step then give final answer.
    """
    q       = question_data["question"]
    choices = question_data["choices"]

    choices_text = ""
    for i, choice in enumerate(choices):
        choices_text += f"{chr(65+i)}) {choice}\n"

    return (
        "You are an expert in industrial asset management, predictive maintenance, "
        "and sensor-based failure detection.\n\n"
        f"Question: {q}\n\n"
        f"{choices_text}\n"
        "Think step by step. Explain your reasoning briefly, "
        "then give your final answer.\n"
        "End with: 'Final answer: X' where X is A, B, C, or D.\n\n"
        "Response:"
    )

# ============================================================
# EXTRACT ANSWER LETTER FROM RESPONSE
# ============================================================

def extract_answer(text):
    """Extract A/B/C/D from model response."""
    if not text:
        return None

    # Primary: "Final answer: X"
    m = re.search(r'final\s+answer[:\s]+([ABCD])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: "answer is X"
    m = re.search(r'answer\s+is\s+([ABCD])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fallback: last standalone letter in response
    for line in reversed(text.strip().split('\n')):
        line = line.strip()
        if line.upper() in ['A','B','C','D']:
            return line.upper()
        m = re.search(r'\b([ABCD])\b', line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None

# ============================================================
# GENERATION WITH LOGPROBS
# ============================================================

def generate_with_confidence(model_name, prompt, max_tokens=300):
    """
    Generate response and calculate confidence from logprobs.
    Confidence = mean(exp(logprob)) across all generated tokens.
    """
    payload = {
        "model":    model_name,
        "prompt":   prompt,
        "stream":   False,
        "logprobs": True,
        "options":  {"temperature": 0.1, "num_predict": max_tokens}
    }
    if is_qwen(model_name):
        payload["think"] = False

    for attempt in range(2):
        try:
            start = time.time()
            r = requests.post(OLLAMA_GENERATE, json=payload,
                              timeout=get_timeout(model_name))
            r.raise_for_status()
            elapsed = round(time.time() - start, 2)
            data    = r.json()
            text    = data.get("response", "").strip()

            if len(text.split()) < 5 and attempt == 0:
                print(f"  [WARN] Short response, retrying...")
                time.sleep(3)
                continue

            confidence    = 0.5
            logprobs_data = data.get("logprobs", [])
            if logprobs_data and isinstance(logprobs_data, list):
                lp_values = [
                    ti["logprob"] for ti in logprobs_data
                    if isinstance(ti, dict) and "logprob" in ti
                ]
                if lp_values:
                    confidence = round(math.exp(sum(lp_values)/len(lp_values)), 4)

            return text, confidence, elapsed, len(text.split())

        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt == 0:
                time.sleep(5)

    return "", 0.0, 0.0, 0

# ============================================================
# CONDITION 1 — BASELINE
# Always use Qwen3.5 9B — runs on TEST SET only
# ============================================================

print("=" * 60)
print(f"CONDITION 1 — BASELINE (Always {BASELINE_MODEL})")
print(f"  Test set: {len(EVAL_Q)} questions")
print("=" * 60)

baseline_results = []

for i, q_data in enumerate(EVAL_Q):
    correct_letter = q_data["correct_letter"]
    choices        = q_data["choices"]

    print(f"\n  [{i+1:3d}/{len(EVAL_Q)}]")
    print(f"  QUESTION: {q_data['question']}")
    for j, choice in enumerate(choices):
        print(f"    {chr(65+j)}) {choice}")
    print(f"  Correct answer: {correct_letter}")
    print(f"  ─────────────────────────────────────────────")

    text, conf, lat, tok = generate_with_confidence(
        BASELINE_MODEL, build_cot_prompt(q_data)
    )
    predicted = extract_answer(text)
    correct   = 1 if predicted == correct_letter else 0

    print(f"\n  COT REASONING ({BASELINE_MODEL}):")
    print(f"  {text}")
    print(f"\n  Predicted: {predicted}  → {'✅ CORRECT' if correct else '❌ WRONG'}")
    print(f"  conf={conf}  lat={lat}s")

    baseline_results.append({
        "question":       q_data["question"],
        "choices":        choices,
        "correct_letter": correct_letter,
        "predicted":      predicted,
        "correct":        correct,
        "model":          BASELINE_MODEL,
        "confidence":     conf,
        "latency":        lat,
        "tokens":         tok,
        "cot_reasoning":  text
    })

base_acc = avg([r["correct"] for r in baseline_results])
base_lat = avg([r["latency"] for r in baseline_results])

print(f"\n  ── BASELINE SUMMARY ──")
print(f"  Accuracy   : {base_acc*100:.1f}% "
      f"({sum(r['correct'] for r in baseline_results)}/{len(baseline_results)})")
print(f"  Avg latency: {base_lat}s")

# ============================================================
# CONDITION 2 — AdaptEvolve
# Confidence cascade — strictly ascending sizes
# Runs on TEST SET only
# ============================================================

print("\n" + "=" * 60)
print(f"CONDITION 2 — AdaptEvolve (threshold={CONFIDENCE_THRESHOLD})")
print(f"  Test set: {len(EVAL_Q)} questions")
print(f"  Cascade: " +
      " → ".join(f"{m['params_b']}B" for m in CASCADE_MODELS))
print("=" * 60)

adaptevolve_results = []

for i, q_data in enumerate(EVAL_Q):
    correct_letter = q_data["correct_letter"]
    choices        = q_data["choices"]

    print(f"\n  [{i+1:3d}/{len(EVAL_Q)}] threshold={CONFIDENCE_THRESHOLD}")
    print(f"  QUESTION: {q_data['question']}")
    for j, choice in enumerate(choices):
        print(f"    {chr(65+j)}) {choice}")
    print(f"  Correct answer: {correct_letter}")
    print(f"  ─────────────────────────────────────────────")

    prompt = build_cot_prompt(q_data)

    final_text      = ""
    final_model     = ""
    final_conf      = 0.0
    final_lat       = 0.0
    final_tok       = 0
    final_predicted = None
    final_correct   = 0
    escalations     = 0
    all_confidences = []

    for model_info in CASCADE_MODELS:
        model_name = model_info["name"]
        text, conf, lat, tok = generate_with_confidence(model_name, prompt)

        predicted = extract_answer(text)
        correct   = 1 if predicted == correct_letter else 0

        print(f"    → {model_name:25s} conf={conf:.4f}  lat={lat}s  "
              f"pred={predicted}  {'✅' if correct else '❌'}", end="")

        all_confidences.append({
            "model":      model_name,
            "params_b":   model_info["params_b"],
            "confidence": conf,
            "predicted":  predicted,
            "correct":    correct
        })

        final_text      = text
        final_model     = model_name
        final_conf      = conf
        final_lat      += lat
        final_tok       = tok
        final_predicted = predicted
        final_correct   = correct

        if conf >= CONFIDENCE_THRESHOLD:
            print(f"  ✅ ACCEPTED")
            print(f"\n  COT REASONING ({model_name}):")
            print(f"  {text}")
            break
        else:
            print(f"  ↑ escalate")
            escalations += 1
            if model_name != CASCADE_MODELS[-1]["name"]:
                time.sleep(1)

    print(f"  ── Final: {final_model}  correct={final_correct}  "
          f"lat={round(final_lat,2)}s  escalations={escalations}")

    adaptevolve_results.append({
        "question":        q_data["question"],
        "choices":         choices,
        "correct_letter":  correct_letter,
        "predicted":       final_predicted,
        "correct":         final_correct,
        "final_model":     final_model,
        "final_conf":      final_conf,
        "total_latency":   round(final_lat, 2),
        "tokens":          final_tok,
        "escalations":     escalations,
        "all_confidences": all_confidences,
        "cot_reasoning":   final_text
    })

adapt_acc = avg([r["correct"]       for r in adaptevolve_results])
adapt_lat = avg([r["total_latency"] for r in adaptevolve_results])
esc_rate  = sum(1 for r in adaptevolve_results if r["escalations"] > 0)
model_usage = Counter(r["final_model"] for r in adaptevolve_results)

# Overconfidence: confident >= threshold but WRONG
overconfident = [
    r for r in adaptevolve_results
    if r["final_conf"] >= CONFIDENCE_THRESHOLD and r["correct"] == 0
]

lat_saving = round((base_lat - adapt_lat) / base_lat * 100) if base_lat > 0 else 0

print(f"\n  ── AdaptEvolve SUMMARY ──")
print(f"  Accuracy    : {adapt_acc*100:.1f}% "
      f"({sum(r['correct'] for r in adaptevolve_results)}/{len(adaptevolve_results)})")
print(f"  Avg latency : {adapt_lat}s")
print(f"  Avg escalations: {avg([r['escalations'] for r in adaptevolve_results])}")
print(f"  Model usage:")
for model, count in model_usage.most_common():
    pct = round(count / len(adaptevolve_results) * 100)
    print(f"    {model:30s}: {count:3d} ({pct}%)")
print(f"  Escalation rate: {esc_rate}/{len(adaptevolve_results)} "
      f"({round(esc_rate/len(adaptevolve_results)*100)}%)")
print(f"  Overconfident wrong: {len(overconfident)} cases")

# ============================================================
# PLOTS
# ============================================================

C_BASE  = "#2E5E9E"
C_ADAPT = "#E07B35"

# 1 — Accuracy
fig, ax = plt.subplots(figsize=(7, 5))
vals  = [base_acc*100, adapt_acc*100]
bars  = ax.bar(["Baseline\n(9B always)", "AdaptEvolve\n(cascade)"],
               vals, color=[C_BASE, C_ADAPT], alpha=0.85, width=0.4)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("Experiment 5 — Accuracy: Baseline vs AdaptEvolve\n"
             "(FailureSensorIQ, ground truth evaluation, test set)",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_accuracy.png"), dpi=150)
plt.close()
print(f"\n✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_accuracy.png')}")

# 2 — Latency
fig, ax = plt.subplots(figsize=(7, 5))
vals  = [base_lat, adapt_lat]
bars  = ax.bar(["Baseline\n(9B always)", "AdaptEvolve\n(cascade)"],
               vals, color=[C_BASE, C_ADAPT], alpha=0.85, width=0.4)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title("Experiment 5 — Latency: Baseline vs AdaptEvolve\n"
             "(FailureSensorIQ, test set)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_latency.png"), dpi=150)
plt.close()
print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_latency.png')}")

# 3 — Model usage
fig, ax = plt.subplots(figsize=(8, 5))
labels  = [m["label"] for m in CASCADE_MODELS]
counts  = [model_usage.get(m["name"], 0) for m in CASCADE_MODELS]
colors  = plt.cm.Blues(np.linspace(0.3, 0.9, len(labels)))
bars    = ax.bar(labels, counts, color=colors, alpha=0.85)
for bar, cnt in zip(bars, counts):
    if cnt > 0:
        pct = round(cnt / len(adaptevolve_results) * 100)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{cnt}\n({pct}%)", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Questions Handled", fontsize=11)
ax.set_title("Experiment 5 — Model Usage Distribution (AdaptEvolve)\n"
             "(FailureSensorIQ, test set)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_model_usage.png"), dpi=150)
plt.close()
print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_model_usage.png')}")

# 4 — Confidence vs Correctness scatter
# Key plot: tests whether high confidence = correct answer
# Overconfident wrong answers = high-x red points above threshold line
fig, ax = plt.subplots(figsize=(8, 6))
for r in adaptevolve_results:
    for call in r["all_confidences"]:
        color  = "#2E9E5E" if call["correct"] == 1 else "#E03535"
        marker = "o"       if call["correct"] == 1 else "x"
        ax.scatter(call["confidence"], call["correct"] + np.random.uniform(-0.02, 0.02),
                   color=color, marker=marker, alpha=0.4, s=30)
ax.axvline(x=CONFIDENCE_THRESHOLD, color="orange", linestyle="--",
           linewidth=1.5, label=f"Threshold = {CONFIDENCE_THRESHOLD}")
ax.set_xlabel("Confidence Score (mean token logprob)", fontsize=11)
ax.set_ylabel("Correctness (1=correct, 0=wrong)", fontsize=11)
ax.set_title("Experiment 5 — Confidence vs Correctness\n"
             "Green=correct, Red=wrong. High-confidence red points = overconfidence problem",
             fontsize=10, fontweight="bold")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Wrong (0)", "Correct (1)"])
handles = [
    mpatches.Patch(color="#2E9E5E", label="Correct answer"),
    mpatches.Patch(color="#E03535", label="Wrong answer"),
    mpatches.Patch(color="orange",  label=f"Threshold {CONFIDENCE_THRESHOLD}")
]
ax.legend(handles=handles)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_confidence_vs_correctness.png"), dpi=150)
plt.close()
print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_confidence_vs_correctness.png')}")

# 5 — Escalation distribution
fig, ax = plt.subplots(figsize=(7, 5))
esc_counts = Counter(r["escalations"] for r in adaptevolve_results)
esc_labels = [f"{k} esc." for k in sorted(esc_counts)]
esc_vals   = [esc_counts[k] for k in sorted(esc_counts)]
colors     = plt.cm.Oranges(np.linspace(0.3, 0.9, len(esc_vals)))
ax.bar(esc_labels, esc_vals, color=colors, alpha=0.85)
ax.set_ylabel("Number of Questions", fontsize=11)
ax.set_title("Experiment 5 — Escalation Distribution\n"
             "(FailureSensorIQ, test set)",
             fontsize=10, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_escalation_rate.png"), dpi=150)
plt.close()
print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_escalation_rate.png')}")

# ============================================================
# SAVE RESULTS JSON
# ============================================================

results_data = {
    "config": {
        "dataset":               "FailureSensorIQ (ibm-research/FailureSensorIQ)",
        "quality_metric":        "accuracy_ground_truth",
        "prompting":             "chain_of_thought_cot",
        "confidence_threshold":  CONFIDENCE_THRESHOLD,
        "routing":               "pure_confidence_based_no_hardcoding",
        "cascade_order":         "strictly_ascending_by_parameter_count",
        "calibration_questions": len(CALIBRATION_Q),
        "test_questions":        len(EVAL_Q),
        "evaluation_set":        "test_set_only",
        "cascade_models":        [f"{m['name']} ({m['params_b']}B)"
                                  for m in CASCADE_MODELS]
    },
    "baseline": {
        "model":           BASELINE_MODEL,
        "accuracy":        base_acc,
        "avg_latency":     base_lat,
        "correct_count":   sum(r["correct"] for r in baseline_results),
        "total_questions": len(baseline_results),
        "results":         baseline_results
    },
    "adaptevolve": {
        "accuracy":             adapt_acc,
        "avg_latency":          adapt_lat,
        "correct_count":        sum(r["correct"] for r in adaptevolve_results),
        "total_questions":      len(adaptevolve_results),
        "escalation_rate":      f"{esc_rate}/{len(adaptevolve_results)}",
        "latency_saving_pct":   lat_saving,
        "overconfident_errors": len(overconfident),
        "model_usage":          dict(model_usage),
        "results":              adaptevolve_results
    }
}

results_file = os.path.join(PLOTS_DIR, "exp5_results.json")
with open(results_file, "w") as f:
    json.dump(results_data, f, indent=2)
print(f"✅ Saved: {results_file}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 5 COMPLETE")
print("=" * 60)
print(f"\nDATASET    : FailureSensorIQ (ground truth evaluation)")
print(f"PROMPTING  : Chain of Thought (CoT)")
print(f"EVAL SET   : Test set ({len(EVAL_Q)} questions)")
print(f"\nKEY RESULTS:")
print(f"  Baseline  accuracy : {base_acc*100:.1f}%")
print(f"  AdaptEvolve accuracy: {adapt_acc*100:.1f}%")
print(f"  Baseline  latency  : {base_lat}s")
print(f"  AdaptEvolve latency: {adapt_lat}s")
print(f"  Latency saving     : ~{lat_saving}%")
print(f"  Escalation rate    : {esc_rate}/{len(adaptevolve_results)} "
      f"({round(esc_rate/len(adaptevolve_results)*100)}%)")
print(f"  Overconfident wrong: {len(overconfident)} cases")
print(f"  (model confident >= {CONFIDENCE_THRESHOLD} but gave wrong answer)")