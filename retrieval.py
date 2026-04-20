import requests
import time
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL  = "http://localhost:11434/api/generate"

# Save plots next to this script file — fixes the "can't find plots" problem
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_DIR}\n")

# Models ordered small → large (order matters for the graph)
MODELS = [
    {"name": "tinyllama:1.1b", "label": "TinyLlama 1.1B", "params_b": 1.1},
    {"name": "phi:latest",     "label": "Phi 2.7B",        "params_b": 2.7},
    {"name": "mistral:latest", "label": "Mistral 7B",      "params_b": 7.0},
]

# ============================================================
# TASKS
# Real battery verification engineering tasks.
# expected_keywords: domain concepts a correct plan must mention.
# ============================================================

TASKS = [
    {
        "task": "Diagnose why a battery pack overheats during high-rate discharge and propose a fix.",
        "expected_keywords": ["temperature", "thermal", "cooling", "discharge", "BMS"]
    },
    {
        "task": "Design a verification test plan for a battery cell's cycle life performance.",
        "expected_keywords": ["charge", "discharge", "cycles", "capacity", "degradation"]
    },
    {
        "task": "Identify root cause of voltage imbalance across cells in a battery module.",
        "expected_keywords": ["voltage", "cell", "balance", "resistance", "BMS"]
    },
    {
        "task": "Plan a safety validation procedure for a battery management system (BMS).",
        "expected_keywords": ["overvoltage", "overcurrent", "temperature", "protection", "test"]
    },
    {
        "task": "Evaluate the impact of low temperature on battery capacity and suggest mitigations.",
        "expected_keywords": ["temperature", "capacity", "electrolyte", "heating", "lithium"]
    },
]

# ============================================================
# MEMORY STORE  (BudgetMem-inspired)
# Accumulates solved task plans, retrieves the most relevant
# ones by keyword overlap before each new task.
# No external vector DB needed.
# ============================================================

memory_store = []

def add_to_memory(task, plan_text):
    keywords = set(task.lower().split() + plan_text.lower().split())
    memory_store.append({"task": task, "plan": plan_text, "keywords": keywords})

def retrieve_from_memory(task, top_k=2):
    if not memory_store:
        return []
    query_words = set(task.lower().split())
    scored = [(len(query_words & e["keywords"]), e) for e in memory_store]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for score, e in scored[:top_k] if score > 0]

# ============================================================
# GENERATION
# ============================================================

def generate(model_name, prompt, max_tokens=250):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens}
    }
    start = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return "", 0.0, 0
    elapsed = round(time.time() - start, 2)
    text = r.json().get("response", "").strip()
    return text, elapsed, len(text.split())

# ============================================================
# PROMPTS
# Memory is injected as a compact note ABOVE a clear separator.
# The model is instructed to write only after "PLAN:" —
# this prevents it from echoing back the memory block.
# ============================================================

def prompt_no_memory(task):
    return (
        "You are a battery verification engineer.\n"
        "Generate a numbered step-by-step technical plan.\n"
        "Rules: numbered steps only | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\n"
        "PLAN:\n"
    )

def prompt_with_memory(task):
    relevant = retrieve_from_memory(task)
    if not relevant:
        return prompt_no_memory(task)

    # Compact memory: only task titles + first 3 numbered steps
    mem_lines = ["[MEMORY — past solutions for reference]"]
    for i, entry in enumerate(relevant, 1):
        numbered = [l.strip() for l in entry["plan"].split("\n")
                    if l.strip() and l.strip()[0].isdigit()][:3]
        mem_lines.append(f"Ref {i}: {entry['task']}")
        if numbered:
            mem_lines.append("Steps: " + " | ".join(numbered))
    mem_block = "\n".join(mem_lines)

    return (
        "You are a battery verification engineer.\n\n"
        f"{mem_block}\n\n"
        "Using the memory above as guidance where relevant, "
        "generate a numbered step-by-step technical plan for the task below.\n"
        "Rules: numbered steps only | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\n"
        "PLAN:\n"
    )

# ============================================================
# OUTPUT CLEANING
# Extracts only the numbered plan lines from model output.
# This removes any leaked prompt text, memory blocks, or drift.
# ============================================================

def extract_plan(text):
    lines = text.split("\n")
    plan_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Accept lines that start with a digit or "Step"
        if s[0].isdigit() or s.lower().startswith("step"):
            plan_lines.append(s)
    # Fallback: nothing matched, return cleaned original
    if not plan_lines:
        return text.strip()
    return "\n".join(plan_lines)

# ============================================================
# SCORING
# step_coverage — deterministic keyword match (reliable)
# latency       — measured directly during generation
#
# NOTE: Correctness and coherence were removed because the
# local judge model (mistral) was not strong enough to produce
# differentiated scores — all values collapsed to the same
# default. Step coverage is deterministic and fully reliable.
# ============================================================

def score_plan(plan_text, expected_keywords):
    """Returns step_coverage (0.0–1.0) via deterministic keyword match."""
    plan_lower = plan_text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in plan_lower)
    return round(matched / len(expected_keywords), 2)

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

results = {}

for model_info in MODELS:
    model_name = model_info["name"]

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    memory_store.clear()  # fresh memory per model — fair comparison

    model_results = {k: [] for k in [
        "coverage_no_mem", "coverage_mem",
        "latency_no_mem",  "latency_mem",
    ]}

    for t in TASKS:
        task     = t["task"]
        keywords = t["expected_keywords"]

        print(f"\n  TASK: {task}")

        # ---- WITHOUT MEMORY ----
        raw1, lat1, _ = generate(model_name, prompt_no_memory(task))
        plan1 = extract_plan(raw1)
        sc1   = score_plan(plan1, keywords)

        print(f"\n  --- WITHOUT MEMORY ---")
        for line in plan1.split("\n"):
            print(f"    {line}")
        print(f"  → coverage={sc1}  latency={lat1}s")

        # ---- WITH MEMORY ----
        raw2, lat2, _ = generate(model_name, prompt_with_memory(task))
        plan2 = extract_plan(raw2)
        sc2   = score_plan(plan2, keywords)

        print(f"\n  --- WITH MEMORY ---")
        for line in plan2.split("\n"):
            print(f"    {line}")
        print(f"  → coverage={sc2}  latency={lat2}s")

        # Store clean plan in memory for subsequent tasks
        add_to_memory(task, plan2)

        model_results["coverage_no_mem"].append(sc1)
        model_results["coverage_mem"].append(sc2)
        model_results["latency_no_mem"].append(lat1)
        model_results["latency_mem"].append(lat2)

    results[model_name] = model_results

    def avg(lst): return round(sum(lst) / len(lst), 3)
    print(f"\n  ── KPI SUMMARY ──")
    print(f"  Step coverage: {avg(model_results['coverage_no_mem'])} → {avg(model_results['coverage_mem'])}")
    print(f"  Latency (avg): {avg(model_results['latency_no_mem'])}s → {avg(model_results['latency_mem'])}s")

# ============================================================
# PLOTTING
# ============================================================

def avg(lst): return round(sum(lst) / len(lst), 3)

model_names  = [m["name"]  for m in MODELS]
model_labels = [m["label"] for m in MODELS]
C_BASE = "#888780"
C_MEM  = "#185FA5"

# ── Plot 1: Step coverage — model size vs memory effect (main graph) ──
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(MODELS))
w = 0.35

cov_no  = [avg(results[m]["coverage_no_mem"]) for m in model_names]
cov_mem = [avg(results[m]["coverage_mem"])    for m in model_names]

bars1 = ax.bar(x - w/2, cov_no,  w, color=C_BASE, alpha=0.85, label="Baseline (no memory)")
bars2 = ax.bar(x + w/2, cov_mem, w, color=C_MEM,  alpha=0.90, label="Optimized (with memory)")

# Annotate values on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=10, color="#444441")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=10, color=C_MEM)

# Annotate delta above each pair
for i, (v1, v2) in enumerate(zip(cov_no, cov_mem)):
    delta = v2 - v1
    color = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
    ax.text(x[i], max(v1, v2) + 0.07, f"{delta:+.2f}",
            ha="center", fontsize=10, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylabel("Step Coverage Score (0–1)", fontsize=11)
ax.set_title("Memory Optimization — Step Coverage by Model Size", fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "step_coverage_by_model.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved: {path1}")

# ── Plot 2: Step coverage per task per model (detailed breakdown) ──
fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 4), sharey=True)
task_labels_short = [f"T{i+1}" for i in range(len(TASKS))]
x_pos = np.arange(len(TASKS))

for ax, model_info in zip(axes, MODELS):
    m = model_info["name"]
    no_mem   = results[m]["coverage_no_mem"]
    with_mem = results[m]["coverage_mem"]

    ax.bar(x_pos - 0.2, no_mem,   0.35, color=C_BASE, alpha=0.85, label="Baseline")
    ax.bar(x_pos + 0.2, with_mem, 0.35, color=C_MEM,  alpha=0.90, label="With memory")

    for i, (v1, v2) in enumerate(zip(no_mem, with_mem)):
        delta = v2 - v1
        col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
        ax.text(x_pos[i], max(v1, v2) + 0.05, f"{delta:+.2f}",
                ha="center", fontsize=8, color=col, fontweight="bold")

    ax.set_title(model_info["label"], fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_labels_short, fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.grid(axis="y", alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel("Step Coverage (0–1)", fontsize=9)
        ax.legend(fontsize=8)

# Task legend below chart
task_legend = "  ".join([f"T{i+1}: {t['task'][:45]}..." for i, t in enumerate(TASKS)])
fig.text(0.01, -0.04, task_legend, fontsize=7, color="#5F5E5A", family="monospace")
fig.suptitle("Step Coverage per Task — Baseline vs Memory Optimized", fontsize=12)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "step_coverage_per_task.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path2}")

# ── Plot 3: Latency comparison ──
fig, ax = plt.subplots(figsize=(9, 5))

lat_no  = [avg(results[m]["latency_no_mem"]) for m in model_names]
lat_mem = [avg(results[m]["latency_mem"])    for m in model_names]

bars1 = ax.bar(x - w/2, lat_no,  w, color=C_BASE, alpha=0.85, label="Baseline (no memory)")
bars2 = ax.bar(x + w/2, lat_mem, w, color=C_MEM,  alpha=0.90, label="Optimized (with memory)")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}s", ha="center", fontsize=10, color="#444441")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}s", ha="center", fontsize=10, color=C_MEM)

# Annotate latency increase
for i, (v1, v2) in enumerate(zip(lat_no, lat_mem)):
    pct = round((v2 - v1) / v1 * 100) if v1 > 0 else 0
    ax.text(x[i], max(v1, v2) + 2.5, f"+{pct}%",
            ha="center", fontsize=10, color="#A32D2D", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title("Latency Trade-off — Baseline vs Memory Optimized", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "latency_comparison.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path3}")

# ── Save raw results JSON ──
serializable = {m: {k: [float(x) for x in v] for k, v in r.items()}
                for m, r in results.items()}
path_json = os.path.join(PLOTS_DIR, "results.json")
with open(path_json, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"✅ Saved: {path_json}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)