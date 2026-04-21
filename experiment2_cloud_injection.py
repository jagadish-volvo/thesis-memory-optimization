"""
Experiment 2 — Cloud Memory Injection
======================================
Step 1: Run 397B-cloud on all 5 tasks → save high-quality plans to memory file
Step 2: Inject that cloud memory into each small model
Step 3: Small models run the same 5 tasks WITH the cloud memory
Step 4: Compare — can small models perform like the large model?

This directly answers the supervisor's question:
"Use the cloud model to run a task, store that in memory,
then inject that memory in the small models to check performances."
"""

import requests
import time
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/chat"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR   = os.path.join(SCRIPT_DIR, "plots_exp2")
MEMORY_DIR  = os.path.join(SCRIPT_DIR, "memory_files")
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}")
print(f"Memory → {MEMORY_DIR}\n")

CLOUD_MODEL = "qwen3.5:397b-cloud"
CLOUD_MEMORY_FILE = os.path.join(MEMORY_DIR, "memory_cloud_397b.json")

# Small models to test with cloud memory injected
SMALL_MODELS = [
    {"name": "tinyllama:1.1b",     "label": "TinyLlama\n1.1B",  "params_b": 1.1},
    {"name": "phi:latest",         "label": "Phi\n2.7B",         "params_b": 2.7},
    {"name": "mistral:latest",     "label": "Mistral\n7B",       "params_b": 7.0},
    {"name": "qwen3.5:0.8b",       "label": "Qwen3.5\n0.8B",    "params_b": 0.8},
    {"name": "qwen3.5:2b",         "label": "Qwen3.5\n2B",      "params_b": 2.0},
    {"name": "qwen3.5:4b",         "label": "Qwen3.5\n4B",      "params_b": 4.0},
    {"name": "qwen3.5:9b",         "label": "Qwen3.5\n9B",      "params_b": 9.0},
]

# ============================================================
# TASKS
# ============================================================

TASKS = [
    {
        "task": "Diagnose why a battery pack overheats during high-rate discharge and propose a fix.",
        "expected_keywords": [
            ["temperature", "thermal", "heat", "temp"],
            ["cooling", "cooler", "cool", "dissipation"],
            ["discharge", "discharging", "c-rate"],
            ["bms", "battery management", "management system"],
            ["resistance", "impedance", "internal resistance"]
        ]
    },
    {
        "task": "Design a verification test plan for a battery cell's cycle life performance.",
        "expected_keywords": [
            ["charge", "charging"],
            ["discharge", "discharging"],
            ["cycle", "cycling", "cycles"],
            ["capacity", "degradation", "fade", "retention"],
            ["voltage", "current", "soc", "state of charge"]
        ]
    },
    {
        "task": "Identify root cause of voltage imbalance across cells in a battery module.",
        "expected_keywords": [
            ["voltage", "imbalance", "volt"],
            ["cell", "cells"],
            ["balance", "balancing", "balanced"],
            ["resistance", "impedance", "internal resistance"],
            ["bms", "battery management", "inspection"]
        ]
    },
    {
        "task": "Plan a safety validation procedure for a battery management system (BMS).",
        "expected_keywords": [
            ["overvoltage", "over-voltage", "over voltage", "voltage limit", "overcharge"],
            ["overcurrent", "over-current", "over current", "current limit"],
            ["temperature", "thermal", "overtemperature", "thermal runaway"],
            ["protection", "protect", "safety", "fault", "isolation"],
            ["test", "testing", "validation", "verify", "validate"]
        ]
    },
    {
        "task": "Evaluate the impact of low temperature on battery capacity and suggest mitigations.",
        "expected_keywords": [
            ["temperature", "thermal", "cold", "low temp", "sub-zero"],
            ["capacity", "degradation", "performance", "retention"],
            ["electrolyte", "chemistry", "lithium", "electrode", "impedance"],
            ["heating", "heater", "warm", "thermal management", "preconditioning"],
            ["mitigation", "strategy", "solution", "insulation", "algorithm"]
        ]
    },
]

# ============================================================
# GENERATION
# ============================================================

def get_timeout(model_name):
    if "cloud"   in model_name: return 120
    elif "9b"    in model_name: return 180
    elif "4b"    in model_name: return 120
    elif "mistral" in model_name or "7b" in model_name.lower(): return 120
    else:                       return 90

def is_qwen(model_name):
    return "qwen" in model_name.lower()

def generate(model_name, prompt, max_tokens=300):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens}
    }
    if is_qwen(model_name):
        payload["think"] = False

    timeout = get_timeout(model_name)

    for attempt in range(2):
        start = time.time()
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            elapsed = round(time.time() - start, 2)
            data = r.json()
            text = data.get("message", {}).get("content", "").strip()
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if text:
                return text, elapsed, len(text.split())
            print(f"  [WARN] Empty response attempt {attempt+1}, retrying...")
        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt == 0:
                time.sleep(5)
    return "", 0.0, 0

# ============================================================
# OUTPUT CLEANING
# ============================================================

def extract_plan(text):
    if not text:
        return ""
    skip = [
        "analyze the request", "determine the plan", "determine the content",
        "drafting", "evaluate constraints", "thinking process",
        "numbered steps only", "max 6 steps", "no extra text", "wait,",
        "actually,", "let me", "refine steps", "refining",
        "analyze the task", "analyze the memory", "review memory", "finalizing",
    ]
    lines = text.split("\n")
    plan_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if any(p in s.lower() for p in skip):
            continue
        if (s[0].isdigit() or s.lower().startswith("step") or
                s.startswith("-") or s.startswith("•")):
            plan_lines.append(s)
    if plan_lines:
        return "\n".join(plan_lines)
    return "\n".join(l.strip() for l in lines if l.strip())

# ============================================================
# SCORING
# ============================================================

def score_plan(plan_text, expected_keywords):
    if not plan_text:
        return 0.0
    plan_lower = plan_text.lower()
    matched = sum(
        1 for group in expected_keywords
        if any(kw.lower() in plan_lower for kw in group)
    )
    return round(matched / len(expected_keywords), 2)

def avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else 0.0

# ============================================================
# STEP 1 — Generate cloud memory
# Run 397B-cloud on all 5 tasks and save plans to disk
# ============================================================

def generate_cloud_memory():
    """
    Runs 397B-cloud on all 5 tasks and saves the high-quality
    plans to a JSON memory file. This is the 'teacher' memory
    that will be injected into small models.
    """
    print("\n" + "#" * 60)
    print(f"STEP 1 — Generating cloud memory using {CLOUD_MODEL}")
    print("#" * 60)

    cloud_memory = []

    for t in TASKS:
        task = t["task"]
        print(f"\n  TASK: {task}")

        prompt = (
            "You are a senior battery verification engineer with deep expertise.\n"
            "Generate a precise, detailed numbered step-by-step technical plan.\n"
            "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
            f"TASK: {task}\n\nPLAN:\n1."
        )

        raw, lat, _ = generate(CLOUD_MODEL, prompt, max_tokens=400)
        plan = extract_plan(raw)

        print(f"  Plan ({lat}s):")
        for line in plan.split("\n")[:6]:
            print(f"    {line}")

        cloud_memory.append({
            "task":     task,
            "plan":     plan,
            "keywords": list(set(task.lower().split() + plan.lower().split())),
            "model":    CLOUD_MODEL,
            "latency":  lat
        })

    # Save to disk
    with open(CLOUD_MEMORY_FILE, "w") as f:
        json.dump(cloud_memory, f, indent=2)
    print(f"\n✅ Cloud memory saved to: {CLOUD_MEMORY_FILE}")
    return cloud_memory

def load_cloud_memory():
    """Load existing cloud memory from disk if available."""
    if os.path.exists(CLOUD_MEMORY_FILE):
        with open(CLOUD_MEMORY_FILE) as f:
            data = json.load(f)
        print(f"✅ Loaded existing cloud memory ({len(data)} entries) from {CLOUD_MEMORY_FILE}")
        return data
    return None

# ============================================================
# STEP 2 — Build injection prompt using cloud memory
# ============================================================

def build_cloud_injection_prompt(task, cloud_memory):
    """
    Builds a prompt that injects the 397B cloud model's high-quality
    solution for the most relevant past task as memory context.
    """
    # Find most relevant cloud memory entry by keyword overlap
    query_words = set(task.lower().split())
    best_entry  = None
    best_score  = 0

    for entry in cloud_memory:
        entry_words = set(entry["task"].lower().split() + entry["plan"].lower().split())
        score = len(query_words & entry_words)
        if score > best_score:
            best_score = score
            best_entry = entry

    if not best_entry or best_score == 0:
        # Fallback: no memory
        return (
            "You are a battery verification engineer.\n"
            "Generate a numbered step-by-step technical plan.\n"
            "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
            f"TASK: {task}\n\nPLAN:\n1."
        )

    # Inject cloud model's solution as reference
    ref_steps = "\n".join(
        [l.strip() for l in best_entry["plan"].split("\n")
         if l.strip() and l.strip()[0].isdigit()][:4]
    )

    return (
        "You are a battery verification engineer.\n\n"
        f"[EXPERT REFERENCE — high quality solution from advanced model]\n"
        f"Reference task: {best_entry['task']}\n"
        f"Reference steps:\n{ref_steps}\n\n"
        "Using the expert reference above as guidance, generate a numbered "
        "step-by-step technical plan for the task below.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\nPLAN:\n1."
    )

def prompt_no_memory(task):
    return (
        "You are a battery verification engineer.\n"
        "Generate a numbered step-by-step technical plan.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\nPLAN:\n1."
    )

# ============================================================
# STEP 3 — Run small models with cloud memory injected
# ============================================================

def run_small_models_with_cloud_memory(cloud_memory):
    print("\n" + "#" * 60)
    print("STEP 2 — Running small models with cloud memory injected")
    print("#" * 60)

    results = {}

    for model_info in SMALL_MODELS:
        model_name = model_info["name"]

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 60}")

        model_results = {k: [] for k in [
            "coverage_baseline",    # no memory at all
            "coverage_cloud_mem",   # with 397B cloud memory injected
            "latency_baseline",
            "latency_cloud_mem",
        ]}

        for t in TASKS:
            task     = t["task"]
            keywords = t["expected_keywords"]

            print(f"\n  TASK: {task}")

            # BASELINE — no memory
            raw1, lat1, _ = generate(model_name, prompt_no_memory(task))
            plan1 = extract_plan(raw1)
            sc1   = score_plan(plan1, keywords)
            print(f"\n  --- BASELINE (no memory) ---")
            for line in plan1.split("\n")[:5]:
                print(f"    {line}")
            print(f"  → coverage={sc1}  latency={lat1}s")

            # WITH CLOUD MEMORY
            cloud_prompt = build_cloud_injection_prompt(task, cloud_memory)
            raw2, lat2, _ = generate(model_name, cloud_prompt)
            plan2 = extract_plan(raw2)
            sc2   = score_plan(plan2, keywords)
            print(f"\n  --- WITH 397B CLOUD MEMORY ---")
            for line in plan2.split("\n")[:5]:
                print(f"    {line}")
            print(f"  → coverage={sc2}  latency={lat2}s")

            model_results["coverage_baseline"].append(sc1)
            model_results["coverage_cloud_mem"].append(sc2)
            model_results["latency_baseline"].append(lat1)
            model_results["latency_cloud_mem"].append(lat2)

        results[model_name] = model_results
        print(f"\n  ── KPI SUMMARY ──")
        print(f"  Step coverage: {avg(model_results['coverage_baseline'])} → {avg(model_results['coverage_cloud_mem'])}")
        print(f"  Latency (avg): {avg(model_results['latency_baseline'])}s → {avg(model_results['latency_cloud_mem'])}s")

    return results

# ============================================================
# PLOTTING
# ============================================================

def plot_results(results, cloud_memory):
    model_names  = [m["name"]  for m in SMALL_MODELS]
    model_labels = [m["label"] for m in SMALL_MODELS]

    # Cloud model reference score
    cloud_scores = []
    for t in TASKS:
        entry = next((e for e in cloud_memory if e["task"] == t["task"]), None)
        if entry:
            cloud_scores.append(score_plan(entry["plan"], t["expected_keywords"]))
    cloud_avg = avg(cloud_scores)

    C_BASE  = "#888780"
    C_CLOUD = "#185FA5"
    x = np.arange(len(SMALL_MODELS))
    w = 0.35

    # Plot 1: Step coverage comparison
    fig, ax = plt.subplots(figsize=(14, 5))

    cov_base  = [avg(results[m]["coverage_baseline"])  for m in model_names]
    cov_cloud = [avg(results[m]["coverage_cloud_mem"]) for m in model_names]

    bars1 = ax.bar(x - w/2, cov_base,  w, color=C_BASE,  alpha=0.85, label="Baseline (no memory)")
    bars2 = ax.bar(x + w/2, cov_cloud, w, color=C_CLOUD, alpha=0.90, label="With 397B cloud memory")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9, color="#444441")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9, color=C_CLOUD)

    for i, (v1, v2) in enumerate(zip(cov_base, cov_cloud)):
        delta = v2 - v1
        col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
        ax.text(x[i], max(v1, v2) + 0.07, f"{delta:+.2f}",
                ha="center", fontsize=9, color=col, fontweight="bold")

    # Reference line showing 397B cloud performance
    ax.axhline(y=cloud_avg, color="#993C1D", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(len(SMALL_MODELS) - 0.3, cloud_avg + 0.02,
            f"397B target ({cloud_avg:.2f})", fontsize=9, color="#993C1D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Step Coverage Score (0-1)", fontsize=11)
    ax.set_title("Experiment 2 — Cloud Memory Injection: Can Small Models Match 397B?",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, "exp2_cloud_injection_coverage.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # Plot 2: Improvement from cloud memory
    fig, ax = plt.subplots(figsize=(12, 4))
    improvements = [
        avg(results[m]["coverage_cloud_mem"]) - avg(results[m]["coverage_baseline"])
        for m in model_names
    ]
    colors = ["#0F6E56" if v > 0 else ("#A32D2D" if v < 0 else "#888780")
              for v in improvements]
    bars = ax.bar(x, improvements, 0.5, color=colors, alpha=0.85)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.005 if val >= 0 else -0.02),
                f"{val:+.2f}", ha="center", fontsize=10, fontweight="bold",
                color="#444441")
    ax.axhline(y=0, color="#444441", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Coverage Improvement", fontsize=11)
    ax.set_title("Experiment 2 — Coverage Improvement from 397B Cloud Memory Injection",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "exp2_improvement.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # Save JSON
    serializable = {
        m: {k: [float(v) for v in vals] for k, vals in r.items()}
        for m, r in results.items()
    }
    serializable["cloud_reference"] = {
        "model": CLOUD_MODEL,
        "avg_coverage": float(cloud_avg),
        "per_task": [float(s) for s in cloud_scores]
    }
    path_json = os.path.join(PLOTS_DIR, "exp2_results.json")
    with open(path_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Saved: {path_json}")

# ============================================================
# MAIN
# ============================================================

# Load existing cloud memory or generate fresh
cloud_memory = load_cloud_memory()
if not cloud_memory:
    cloud_memory = generate_cloud_memory()

# Run small models with cloud memory
results = run_small_models_with_cloud_memory(cloud_memory)

# Plot results
plot_results(results, cloud_memory)

print("\n" + "=" * 60)
print("EXPERIMENT 2 COMPLETE")
print("=" * 60)
