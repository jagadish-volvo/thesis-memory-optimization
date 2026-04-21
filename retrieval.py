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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_DIR}\n")

# Same model family (qwen3.5), only size changes — clean scientific comparison
MODELS = [
    {"name": "qwen3.5:0.8b",       "label": "Qwen3.5\n0.8B",  "params_b": 0.8},
    {"name": "qwen3.5:2b",         "label": "Qwen3.5\n2B",    "params_b": 2.0},
    {"name": "qwen3.5:4b",         "label": "Qwen3.5\n4B",    "params_b": 4.0},
    {"name": "qwen3.5:9b",         "label": "Qwen3.5\n9B",    "params_b": 9.0},
    {"name": "qwen3.5:397b-cloud", "label": "Qwen3.5\n397B",  "params_b": 397.0},
]

# ============================================================
# TASKS
# Flexible keyword groups — any synonym counts as a match
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
# MEMORY STORE
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
# - think=False disables qwen3.5 thinking mode for clean output
# - Timeout varies by model size
# - Retries once on empty response
# ============================================================

def get_timeout(model_name):
    if "cloud" in model_name:
        return 120
    elif "9b" in model_name:
        return 180
    elif "4b" in model_name:
        return 120
    else:
        return 90

def generate(model_name, prompt, max_tokens=300):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,      # disable thinking mode — clean fast output
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens
        }
    }
    timeout = get_timeout(model_name)

    for attempt in range(2):  # retry once on failure
        start = time.time()
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            elapsed = round(time.time() - start, 2)
            data = r.json()
            text = data.get("message", {}).get("content", "").strip()
            # Strip any residual think blocks
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            if text:
                return text, elapsed, len(text.split())
            else:
                print(f"  [WARN] Empty response on attempt {attempt+1}, retrying...")
        except Exception as e:
            print(f"  [ERROR] attempt {attempt+1}: {e}")
            if attempt == 0:
                print(f"  [INFO] Retrying in 5 seconds...")
                time.sleep(5)

    return "", 0.0, 0

# ============================================================
# PROMPTS
# ============================================================

def prompt_no_memory(task):
    return (
        "You are a battery verification engineer.\n"
        "Generate a numbered step-by-step technical plan.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\n"
        "PLAN:\n"
        "1."
    )

def prompt_with_memory(task):
    relevant = retrieve_from_memory(task)
    if not relevant:
        return prompt_no_memory(task)

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
        "Using the memory above as guidance, "
        "generate a numbered step-by-step technical plan for the task below.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\n"
        "PLAN:\n"
        "1."
    )

# ============================================================
# OUTPUT CLEANING
# ============================================================

def extract_plan(text):
    if not text:
        return ""

    skip_patterns = [
        "analyze the request", "determine the plan", "determine the content",
        "drafting the content", "drafting the steps", "evaluate constraints",
        "constraint check", "thinking process", "numbered steps only",
        "max 6 steps", "no extra text", "wait,", "actually,", "let me",
        "need to add more", "refine steps", "refining", "analyze the task",
        "analyze the memory", "review memory", "review the memory",
        "draft the steps", "finalizing", "drafting the plan",
    ]

    lines = text.split("\n")
    plan_lines = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if any(p in s.lower() for p in skip_patterns):
            continue
        if (s[0].isdigit() or
            s.lower().startswith("step") or
            s.startswith("-") or
            s.startswith("•")):
            plan_lines.append(s)

    if plan_lines:
        return "\n".join(plan_lines)
    return "\n".join(l.strip() for l in lines if l.strip())

# ============================================================
# SCORING — flexible keyword groups
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

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

def avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else 0.0

results = {}

for model_info in MODELS:
    model_name = model_info["name"]

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    memory_store.clear()

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
        for line in plan1.split("\n")[:6]:
            print(f"    {line}")
        print(f"  → coverage={sc1}  latency={lat1}s")

        # ---- WITH MEMORY ----
        raw2, lat2, _ = generate(model_name, prompt_with_memory(task))
        plan2 = extract_plan(raw2)
        sc2   = score_plan(plan2, keywords)

        print(f"\n  --- WITH MEMORY ---")
        for line in plan2.split("\n")[:6]:
            print(f"    {line}")
        print(f"  → coverage={sc2}  latency={lat2}s")

        add_to_memory(task, plan2)

        model_results["coverage_no_mem"].append(sc1)
        model_results["coverage_mem"].append(sc2)
        model_results["latency_no_mem"].append(lat1)
        model_results["latency_mem"].append(lat2)

    results[model_name] = model_results

    print(f"\n  ── KPI SUMMARY ──")
    print(f"  Step coverage: {avg(model_results['coverage_no_mem'])} → {avg(model_results['coverage_mem'])}")
    print(f"  Latency (avg): {avg(model_results['latency_no_mem'])}s → {avg(model_results['latency_mem'])}s")

# ============================================================
# PLOTTING
# ============================================================

model_names  = [m["name"]  for m in MODELS]
model_labels = [m["label"] for m in MODELS]
C_BASE = "#888780"
C_MEM  = "#185FA5"
x = np.arange(len(MODELS))
w = 0.35

# ── Plot 1: Step coverage main graph ──
fig, ax = plt.subplots(figsize=(12, 5))

cov_no  = [avg(results[m]["coverage_no_mem"]) for m in model_names]
cov_mem = [avg(results[m]["coverage_mem"])    for m in model_names]

bars1 = ax.bar(x - w/2, cov_no,  w, color=C_BASE, alpha=0.85, label="Baseline (no memory)")
bars2 = ax.bar(x + w/2, cov_mem, w, color=C_MEM,  alpha=0.90, label="Optimized (with memory)")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=10, color="#444441")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=10, color=C_MEM)

for i, (v1, v2) in enumerate(zip(cov_no, cov_mem)):
    delta = v2 - v1
    col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
    ax.text(x[i], max(v1, v2) + 0.07, f"{delta:+.2f}",
            ha="center", fontsize=10, color=col, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylabel("Step Coverage Score (0-1)", fontsize=11)
ax.set_title("Memory Optimization — Step Coverage by Model Size (Qwen3.5 Family)",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "step_coverage_by_model.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved: {path1}")

# ── Plot 2: Step coverage per task per model ──
fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 4), sharey=True)
x_pos = np.arange(len(TASKS))
task_labels_short = [f"T{i+1}" for i in range(len(TASKS))]

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

    ax.set_title(model_info["label"].replace("\n", " "), fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_labels_short, fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.grid(axis="y", alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel("Step Coverage (0-1)", fontsize=9)
        ax.legend(fontsize=8)

task_legend = "  ".join([f"T{i+1}: {t['task'][:42]}..." for i, t in enumerate(TASKS)])
fig.text(0.01, -0.04, task_legend, fontsize=7, color="#5F5E5A", family="monospace")
fig.suptitle("Step Coverage per Task — Baseline vs Memory Optimized (Qwen3.5)", fontsize=12)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "step_coverage_per_task.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path2}")

# ── Plot 3: Latency comparison ──
fig, ax = plt.subplots(figsize=(12, 5))

lat_no  = [avg(results[m]["latency_no_mem"]) for m in model_names]
lat_mem = [avg(results[m]["latency_mem"])    for m in model_names]

bars1 = ax.bar(x - w/2, lat_no,  w, color=C_BASE, alpha=0.85, label="Baseline (no memory)")
bars2 = ax.bar(x + w/2, lat_mem, w, color=C_MEM,  alpha=0.90, label="Optimized (with memory)")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}s", ha="center", fontsize=9, color="#444441")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}s", ha="center", fontsize=9, color=C_MEM)

for i, (v1, v2) in enumerate(zip(lat_no, lat_mem)):
    if v1 > 0:
        pct = round((v2 - v1) / v1 * 100)
        sign = "+" if pct >= 0 else ""
        col = "#A32D2D" if pct > 5 else ("#0F6E56" if pct < -5 else "#888780")
        ax.text(x[i], max(v1, v2) + 1.5, f"{sign}{pct}%",
                ha="center", fontsize=9, color=col, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title("Latency Trade-off — Baseline vs Memory Optimized (Qwen3.5)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "latency_comparison.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path3}")

# ── Save results JSON ──
serializable = {
    m: {k: [float(v) for v in vals] for k, vals in r.items()}
    for m, r in results.items()
}
path_json = os.path.join(PLOTS_DIR, "results.json")
with open(path_json, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"✅ Saved: {path_json}")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)