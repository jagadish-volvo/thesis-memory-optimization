"""
Experiment 5 — AdaptEvolve: Free Reasoning with CoT Quality-Based Routing
==========================================================================
Implements adaptive LLM selection inspired by the AdaptEvolve paper.

KEY CHANGE FROM PREVIOUS VERSION:
  Previous: Multiple choice quiz (A/B/C/D) — model forced to pick a letter
  Current:  Free reasoning — model answers openly without predefined options
  Why:      Forcing A/B/C/D constrains reasoning, causes hallucination,
            and biases the model toward justifying a forced choice rather
            than solving the actual problem (Hafid feedback May 2026)

Escalation signal:
  Previous: Logprob confidence score >= threshold
  Current:  Model self-reports reasoning confidence (High/Medium/Low)
            at the end of its CoT reasoning.
            High   → accept answer, stop cascade
            Medium → escalate to next model
            Low    → escalate to next model
  Why:      Reasoning quality during generation is more meaningful than
            token-level confidence of a forced answer selection.

Dataset: FailureSensorIQ (ibm-research/FailureSensorIQ)
  - Expert-curated benchmark for industrial assets
  - Built from ISO documents covering predictive maintenance,
    sensor fault detection, and failure mode reasoning
  - Questions used WITHOUT answer choices (free reasoning format)

Two conditions compared:

Condition 1 — BASELINE (Fixed large model):
  Always use Qwen3.5 9B. Model reasons freely and gives its own answer.

Condition 2 — AdaptEvolve (CoT quality-based cascade):
  Start with smallest model (Qwen3.5 0.8B).
  Model reasons freely and self-reports confidence at end of reasoning.
  If confidence = High → accept answer, stop cascade.
  If confidence = Medium or Low → escalate to next larger model.
  Cascade strictly ascending: 0.8B → 1.1B → 2B → 2.7B → 4B → 7B → 9B

Calibration / Test split:
  Calibration set (first 100 questions) — used to validate approach
  Test set        (last  100 questions) — results reported here only

KPIs:
  - Reasoning quality distribution : High/Medium/Low per model
  - Escalation rate                : how often reasoning was insufficient
  - Model usage distribution       : which models handled which tasks
  - Average latency                : total time including escalations
  - Overconfidence cases           : High confidence but weak reasoning
"""

import requests
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()
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

# ============================================================
# GROQ JUDGE MODEL CONFIG
# Judge model evaluates reasoning quality independently
# Uses Llama 3.1 70B via Groq API (free tier)
# Replace GROQ_API_KEY with your actual key
# ============================================================

import os as _os

# ============================================================
# JUDGE MODEL — DISABLED FOR NOW
# Network restrictions prevent external API access
# All CoT reasoning saved in JSON for supervisor review
# Supervisor will evaluate reasoning quality as domain expert
# ============================================================

ANTHROPIC_API_KEY = ""
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
JUDGE_MODEL       = "claude-haiku-4-5-20251001"
USE_JUDGE         = False  # disabled

def judge_reasoning(question, reasoning, correct_answer=""):
    return None, "Judge disabled"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots_exp5")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}\n")

# Cascade order — strictly ascending by parameter count
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
# LOAD FAILURESENSORIQ — QUESTIONS ONLY (NO CHOICES)
# ============================================================

def load_failuresensoriq(max_questions=200):
    """
    Load FailureSensorIQ questions WITHOUT answer choices.
    Free reasoning format — model answers openly without predefined options.
    """
    cache_file = os.path.join(SCRIPT_DIR, "failuresensoriq_cache.json")

    if os.path.exists(cache_file):
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file) as f:
            questions = json.load(f)
        # Add correct_answer field if missing from old cache
        for q in questions:
            if "correct_answer" not in q:
                choices = q.get("choices", [])
                letter  = q.get("correct_letter", "A")
                idx     = ord(letter.upper()) - 65
                q["correct_answer"] = choices[idx] \
                    if choices and idx < len(choices) else letter
        print(f"  Loaded {len(questions)} questions from cache")
        return questions[:max_questions]

    print("  Loading from Hugging Face (ibm-research/FailureSensorIQ)...")
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
                "correct_answer": choices[ord(correct_letter) - 65]
                                  if choices and len(choices) > ord(correct_letter) - 65
                                  else correct_letter
            })

        print(f"  Loaded {len(questions)} questions")
        with open(cache_file, "w") as f:
            json.dump(questions, f, indent=2)
        print(f"  Cached to: {cache_file}")
        return questions[:max_questions]

    except Exception as e:
        print(f"  [ERROR] Could not load dataset: {e}")
        exit(1)


print("=" * 60)
print("Loading FailureSensorIQ benchmark dataset...")
print("=" * 60)
ALL_QUESTIONS = load_failuresensoriq(max_questions=200)
print(f"\nTotal questions loaded: {len(ALL_QUESTIONS)}")

# ============================================================
# CALIBRATION / TEST SPLIT
# ============================================================

CALIBRATION_Q = ALL_QUESTIONS[:100]
TEST_Q        = ALL_QUESTIONS[100:]
EVAL_Q        = TEST_Q

print(f"\nCalibration set : {len(CALIBRATION_Q)} questions")
print(f"Test set        : {len(EVAL_Q)} questions (results reported here)")
print()

# ============================================================
# TEST JUDGE MODEL AT STARTUP
# ============================================================
if USE_JUDGE:
    print("Testing Groq judge model connection...")
    test_score, test_label = judge_reasoning(
        "Which sensor detects bearing wear in a gas turbine?",
        "Vibration sensors detect bearing wear because worn bearings "
        "generate specific frequency signatures detectable by accelerometers. "
        "This is the industry standard for rotating machinery. [High]"
    )
    if test_score:
        print(f"  ✅ Judge model working — test score: {test_score}/3 ({test_label})")
    else:
        print(f"  ❌ Judge model error: {test_label}")
        print(f"  Check your GROQ_API_KEY and internet connection")
    print()
else:
    print("  ⚠️  Judge model not configured — set GROQ_API_KEY to enable")
    print()

# ============================================================
# HELPERS
# ============================================================

def get_timeout(model_name):
    if "9b"        in model_name: return 300
    elif "4b"      in model_name: return 180
    elif "mistral" in model_name: return 180
    else:                         return 120

def is_qwen(model_name):
    return "qwen" in model_name.lower()

def avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0

# ============================================================
# FREE REASONING PROMPT — NO ANSWER CHOICES
# ============================================================

def build_free_prompt(question_data):
    """
    Build a free reasoning prompt WITHOUT multiple choice options.

    The model is NOT given A/B/C/D options. It must reason freely
    about the industrial failure scenario and give its own answer.

    This avoids the quiz bias where the model bends its reasoning
    to justify a forced choice rather than solving the actual problem.

    At the end the model self-reports its confidence level so the
    cascade can use reasoning quality as the escalation signal.
    """
    q = question_data["question"]

    return (
        "You are an expert in industrial asset management, predictive "
        "maintenance, and sensor-based failure detection.\n\n"
        f"Question: {q}\n\n"
        "Instructions:\n"
        "1. Think step by step about the failure mechanism.\n"
        "2. Explain which sensor you would prioritize and why.\n"
        "3. Give your final answer clearly.\n"
        "4. At the very end, state your confidence level as exactly "
        "one of: [High] [Medium] [Low]\n\n"
        "Response:"
    )

# ============================================================
# EXTRACT CONFIDENCE LEVEL FROM FREE REASONING
# This is the NEW escalation signal — replaces logprob threshold
# ============================================================

def extract_confidence_level(text):
    """
    Extract the self-reported confidence level from the model response.
    The model is asked to end its response with [High], [Medium] or [Low].

    This is the escalation signal:
      High   → accept answer, stop cascade
      Medium → escalate to next model
      Low    → escalate to next model

    Falls back to Medium if no explicit confidence found.
    """
    if not text:
        return "Low"

    # Look for bracketed confidence at end of response
    matches = re.findall(
        r'\[(High|Medium|Low)\]',
        text,
        re.IGNORECASE
    )
    if matches:
        return matches[-1].capitalize()

    # Fallback: look for confidence keywords near end of response
    last_200 = text[-200:].lower()
    if "high confidence" in last_200 or "highly confident" in last_200:
        return "High"
    elif "low confidence" in last_200 or "not confident" in last_200 \
            or "uncertain" in last_200:
        return "Low"
    elif "medium confidence" in last_200 or "moderate" in last_200:
        return "Medium"

    # Default to Medium — triggers escalation
    return "Medium"

# ============================================================
# GENERATION
# ============================================================

def generate_response(model_name, prompt, max_tokens=600):
    """
    Generate a free reasoning response from the model.
    No logprobs needed — escalation is based on self-reported confidence.
    """
    payload = {
        "model":   model_name,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.1, "num_predict": max_tokens}
    }
    if is_qwen(model_name):
        payload["think"] = False

    for attempt in range(2):
        try:
            start = time.time()
            r = requests.post(
                OLLAMA_GENERATE, json=payload,
                timeout=get_timeout(model_name)
            )
            r.raise_for_status()
            elapsed = round(time.time() - start, 2)
            data    = r.json()
            text    = data.get("response", "").strip()

            if len(text.split()) < 10 and attempt == 0:
                print(f"  [WARN] Short response, retrying...")
                time.sleep(3)
                continue

            return text, elapsed, len(text.split())

        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt == 0:
                time.sleep(5)

    return "", 0.0, 0

# ============================================================
# CONDITION 1 — BASELINE
# Always use Qwen3.5 9B — free reasoning, no choices given
# ============================================================

print("=" * 60)
print(f"CONDITION 1 — BASELINE (Always {BASELINE_MODEL})")
print(f"  Free reasoning — no multiple choice options given")
print(f"  Test set: {len(EVAL_Q)} questions")
print("=" * 60)

baseline_results = []

for i, q_data in enumerate(EVAL_Q):
    print(f"\n  [{i+1:3d}/{len(EVAL_Q)}]")
    print(f"  QUESTION: {q_data['question']}")
    print(f"  Expected answer: {q_data['correct_answer']}")
    print(f"  ─────────────────────────────────────────────")

    text, lat, tok = generate_response(
        BASELINE_MODEL, build_free_prompt(q_data)
    )

    confidence = extract_confidence_level(text)

    print(f"\n  FREE REASONING ({BASELINE_MODEL}):")
    print(f"  {text}")

    # Judge model evaluates reasoning quality
    judge_score, judge_label = judge_reasoning(
        q_data["question"], text, q_data["correct_answer"]
    )
    if judge_score:
        print(f"\n  JUDGE SCORE: {judge_score}/3 — {judge_label}")
    print(f"\n  Self-reported confidence: {confidence}")
    print(f"  lat={lat}s  tok={tok}")

    baseline_results.append({
        "question":        q_data["question"],
        "correct_answer":  q_data["correct_answer"],
        "model":           BASELINE_MODEL,
        "reasoning":       text,
        "confidence":      confidence,
        "judge_score":     judge_score,
        "judge_label":     judge_label,
        "latency":         lat,
        "tokens":          tok
    })

base_lat        = avg([r["latency"] for r in baseline_results])
base_conf_dist  = Counter(r["confidence"] for r in baseline_results)

print(f"\n  ── BASELINE SUMMARY ──")
print(f"  Avg latency : {base_lat}s")
print(f"  Confidence distribution:")
for level in ["High", "Medium", "Low"]:
    print(f"    {level}: {base_conf_dist.get(level, 0)}")

# ============================================================
# CONDITION 2 — AdaptEvolve FREE REASONING
# Escalation based on self-reported confidence (High/Medium/Low)
# No multiple choice — model reasons freely
# ============================================================

print("\n" + "=" * 60)
print(f"CONDITION 2 — AdaptEvolve (Free Reasoning)")
print(f"  Escalation signal: self-reported confidence [High/Medium/Low]")
print(f"  High → accept | Medium/Low → escalate")
print(f"  Test set: {len(EVAL_Q)} questions")
print(f"  Cascade: " +
      " → ".join(f"{m['params_b']}B" for m in CASCADE_MODELS))
print("=" * 60)

adaptevolve_results = []

for i, q_data in enumerate(EVAL_Q):
    print(f"\n  [{i+1:3d}/{len(EVAL_Q)}]")
    print(f"  QUESTION: {q_data['question']}")
    print(f"  Expected answer: {q_data['correct_answer']}")
    print(f"  ─────────────────────────────────────────────")

    prompt = build_free_prompt(q_data)

    final_text   = ""
    final_model  = ""
    final_conf   = ""
    final_lat    = 0.0
    final_tok    = 0
    escalations  = 0
    all_attempts = []

    for model_info in CASCADE_MODELS:
        model_name = model_info["name"]
        text, lat, tok = generate_response(model_name, prompt)
        confidence     = extract_confidence_level(text)

        print(f"    → {model_name:25s} conf={confidence:6s}  "
              f"lat={lat}s", end="")

        all_attempts.append({
            "model":      model_name,
            "params_b":   model_info["params_b"],
            "reasoning":  text,
            "confidence": confidence,
            "latency":    lat,
            "tokens":     tok
        })

        final_text  = text
        final_model = model_name
        final_conf  = confidence
        final_lat  += lat
        final_tok   = tok

        if confidence == "High":
            print(f"  ✅ ACCEPTED")
            break
        else:
            print(f"  ↑ escalate ({confidence})")
            escalations += 1
            if model_name != CASCADE_MODELS[-1]["name"]:
                time.sleep(1)

    print(f"\n  ── Accepted: {final_model}")
    print(f"  Self-reported confidence: {final_conf}")
    print(f"  Total latency: {round(final_lat, 2)}s  "
          f"Escalations: {escalations}")
    print(f"\n  FINAL REASONING:")
    print(f"  {final_text}")

    # Judge model evaluates final accepted reasoning quality
    judge_score, judge_label = judge_reasoning(
        q_data["question"], final_text, q_data["correct_answer"]
    )
    if judge_score:
        print(f"\n  JUDGE SCORE: {judge_score}/3 — {judge_label}")

    adaptevolve_results.append({
        "question":        q_data["question"],
        "correct_answer":  q_data["correct_answer"],
        "final_model":     final_model,
        "final_conf":      final_conf,
        "judge_score":     judge_score,
        "judge_label":     judge_label,
        "total_latency":   round(final_lat, 2),
        "tokens":          final_tok,
        "escalations":     escalations,
        "all_attempts":    all_attempts,
        "final_reasoning": final_text
    })

adapt_lat       = avg([r["total_latency"] for r in adaptevolve_results])
esc_rate        = sum(1 for r in adaptevolve_results if r["escalations"] > 0)
model_usage     = Counter(r["final_model"] for r in adaptevolve_results)
adapt_conf_dist = Counter(r["final_conf"] for r in adaptevolve_results)
lat_saving      = round((base_lat - adapt_lat) / base_lat * 100) \
                  if base_lat > 0 else 0

# Judge score distributions
base_judge_dist  = Counter(
    r["judge_label"] for r in baseline_results
    if r.get("judge_score")
)
adapt_judge_dist = Counter(
    r["judge_label"] for r in adaptevolve_results
    if r.get("judge_score")
)
base_judge_avg  = avg([r["judge_score"] for r in baseline_results
                       if r.get("judge_score")])
adapt_judge_avg = avg([r["judge_score"] for r in adaptevolve_results
                       if r.get("judge_score")])

print(f"\n  ── AdaptEvolve SUMMARY ──")
print(f"  Avg latency     : {adapt_lat}s")
print(f"  Latency saving  : ~{lat_saving}%")
print(f"  Avg escalations : {avg([r['escalations'] for r in adaptevolve_results])}")
print(f"  Escalation rate : {esc_rate}/{len(adaptevolve_results)}")
print(f"  Final confidence distribution:")
for level in ["High", "Medium", "Low"]:
    print(f"    {level}: {adapt_conf_dist.get(level, 0)}")
if USE_JUDGE:
    print(f"  Judge score avg : {adapt_judge_avg}/3")
    print(f"  Judge distribution:")
    for label in ["Good", "Partial", "Poor"]:
        print(f"    {label}: {adapt_judge_dist.get(label, 0)}")
print(f"  Model usage:")
for model, count in model_usage.most_common():
    pct = round(count / len(adaptevolve_results) * 100)
    print(f"    {model:30s}: {count:3d} ({pct}%)")

# ============================================================
# PLOTS
# ============================================================

C_BASE  = "#2E5E9E"
C_ADAPT = "#E07B35"

# 1 — Latency comparison
fig, ax = plt.subplots(figsize=(7, 5))
vals = [base_lat, adapt_lat]
bars = ax.bar(
    ["Baseline\n(9B always)", "AdaptEvolve\n(free reasoning)"],
    vals, color=[C_BASE, C_ADAPT], alpha=0.85, width=0.4
)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title(
    "Experiment 5 — Latency: Baseline vs AdaptEvolve\n"
    "(Free Reasoning, CoT Quality-Based Routing, test set)",
    fontsize=10, fontweight="bold"
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_latency.png"), dpi=150)
plt.close()
print(f"\n✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_latency.png')}")

# 2 — Model usage distribution
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
ax.set_title(
    "Experiment 5 — Model Usage Distribution (AdaptEvolve)\n"
    "(Free Reasoning, CoT Quality-Based Routing)",
    fontsize=10, fontweight="bold"
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_model_usage.png"), dpi=150)
plt.close()
print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_model_usage.png')}")

# 3 — Confidence distribution comparison
fig, ax = plt.subplots(figsize=(8, 5))
levels  = ["High", "Medium", "Low"]
x       = np.arange(len(levels))
w       = 0.35
base_vals  = [base_conf_dist.get(l, 0) for l in levels]
adapt_vals = [adapt_conf_dist.get(l, 0) for l in levels]
ax.bar(x - w/2, base_vals,  w, label="Baseline (9B)",  color=C_BASE,  alpha=0.85)
ax.bar(x + w/2, adapt_vals, w, label="AdaptEvolve",     color=C_ADAPT, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=11)
ax.set_ylabel("Number of Questions", fontsize=11)
ax.set_title(
    "Experiment 5 — Self-Reported Confidence Distribution\n"
    "(Free Reasoning — High = accepted, Medium/Low = escalated)",
    fontsize=10, fontweight="bold"
)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_confidence_dist.png"), dpi=150)
plt.close()
print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_confidence_dist.png')}")

# 4 — Escalation distribution
fig, ax = plt.subplots(figsize=(7, 5))
esc_counts = Counter(r["escalations"] for r in adaptevolve_results)
esc_labels = [f"{k} esc." for k in sorted(esc_counts)]
esc_vals   = [esc_counts[k] for k in sorted(esc_counts)]
colors     = plt.cm.Oranges(np.linspace(0.3, 0.9, len(esc_vals)))
ax.bar(esc_labels, esc_vals, color=colors, alpha=0.85)
ax.set_ylabel("Number of Questions", fontsize=11)
ax.set_title(
    "Experiment 5 — Escalation Distribution\n"
    "(Free Reasoning, CoT Quality-Based Routing)",
    fontsize=10, fontweight="bold"
)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "exp5_escalation_rate.png"), dpi=150)
plt.close()
# 5 — Judge score comparison (if judge was used)
if USE_JUDGE and base_judge_dist and adapt_judge_dist:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels     = ["Good (3)", "Partial (2)", "Poor (1)"]
    judge_keys = ["Good", "Partial", "Poor"]
    x          = np.arange(len(labels))
    w          = 0.35
    base_j  = [base_conf_dist.get(k, 0)  for k in judge_keys]
    adapt_j = [adapt_judge_dist.get(k, 0) for k in judge_keys]
    ax.bar(x - w/2, base_j,  w, label=f"Baseline 9B (avg {base_judge_avg}/3)",
           color=C_BASE,  alpha=0.85)
    ax.bar(x + w/2, adapt_j, w, label=f"AdaptEvolve (avg {adapt_judge_avg}/3)",
           color=C_ADAPT, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_title(
        "Experiment 5 — Judge Model Reasoning Quality\n"
        f"(Groq Llama 3.1 70B judge — Good=3, Partial=2, Poor=1)",
        fontsize=10, fontweight="bold"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "exp5_judge_scores.png"), dpi=150)
    plt.close()
    print(f"✅ Saved: {os.path.join(PLOTS_DIR, 'exp5_judge_scores.png')}")

# ============================================================
# SAVE RESULTS JSON
# ============================================================

results_data = {
    "config": {
        "dataset":           "FailureSensorIQ (ibm-research/FailureSensorIQ)",
        "format":            "free_reasoning_no_multiple_choice",
        "escalation_signal": "self_reported_confidence_High_Medium_Low",
        "routing":           "High=accept  Medium/Low=escalate",
        "judge_model":       GROQ_MODEL if USE_JUDGE else "not configured",
        "judge_provider":    "Groq API (free tier)" if USE_JUDGE else "none",
        "cascade_order":     "strictly_ascending_by_parameter_count",
        "test_questions":    len(EVAL_Q),
        "evaluation_set":    "test_set_only",
        "cascade_models":    [f"{m['name']} ({m['params_b']}B)"
                              for m in CASCADE_MODELS]
    },
    "baseline": {
        "model":            BASELINE_MODEL,
        "avg_latency":      base_lat,
        "confidence_dist":  dict(base_conf_dist),
        "judge_score_avg":  base_judge_avg,
        "judge_dist":       dict(base_judge_dist),
        "total_questions":  len(baseline_results),
        "results":          baseline_results
    },
    "adaptevolve": {
        "avg_latency":      adapt_lat,
        "latency_saving":   f"~{lat_saving}%",
        "escalation_rate":  f"{esc_rate}/{len(adaptevolve_results)}",
        "confidence_dist":  dict(adapt_conf_dist),
        "judge_score_avg":  adapt_judge_avg,
        "judge_dist":       dict(adapt_judge_dist),
        "model_usage":      dict(model_usage),
        "total_questions":  len(adaptevolve_results),
        "results":          adaptevolve_results
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
print("EXPERIMENT 5 COMPLETE — FREE REASONING")
print("=" * 60)
print(f"\nDATASET    : FailureSensorIQ")
print(f"FORMAT     : Free reasoning (no multiple choice)")
print(f"ROUTING    : CoT self-reported confidence [High/Medium/Low]")
print(f"JUDGE      : {GROQ_MODEL if USE_JUDGE else 'Not configured'}")
print(f"EVAL SET   : Test set ({len(EVAL_Q)} questions)")
print(f"\nKEY RESULTS:")
print(f"  Baseline  avg latency  : {base_lat}s")
print(f"  AdaptEvolve avg latency: {adapt_lat}s")
print(f"  Latency saving         : ~{lat_saving}%")
print(f"  Escalation rate        : {esc_rate}/{len(adaptevolve_results)} "
      f"({round(esc_rate/len(adaptevolve_results)*100)}%)")
if USE_JUDGE:
    print(f"\nJUDGE SCORES (Groq Llama 3.1 70B):")
    print(f"  Baseline  avg score  : {base_judge_avg}/3")
    print(f"  AdaptEvolve avg score: {adapt_judge_avg}/3")
print(f"\nCONFIDENCE DISTRIBUTION (AdaptEvolve final answers):")
for level in ["High", "Medium", "Low"]:
    print(f"  {level}: {adapt_conf_dist.get(level, 0)} questions")
print(f"\nMODEL USAGE (AdaptEvolve):")
for model, count in model_usage.most_common():
    pct = round(count / len(adaptevolve_results) * 100)
    print(f"  {model:30s}: {count:3d} ({pct}%)")