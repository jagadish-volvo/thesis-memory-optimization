"""
Experiment 1 — Architecture Comparison
=======================================
Compares 8 models across two families:
  - Mixed family: TinyLlama 1.1B, Phi 2.7B, Mistral 7B
  - Qwen3.5 family: 0.8B, 2B, 4B, 9B, 397B-cloud

Goal: Show how both architecture AND size affect performance,
      with and without keyword-based memory optimization.

KPIs: Step Coverage (deterministic), Latency (measured), Token Count (measured)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots_exp1")
MEMORY_DIR = os.path.join(SCRIPT_DIR, "memory_files")
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}")
print(f"Memory → {MEMORY_DIR}\n")

# All 8 models — mixed family + qwen3.5 family
MODELS = [
    # Mixed family (different architectures)
    {"name": "tinyllama:1.1b",      "label": "TinyLlama\n1.1B",  "params_b": 1.1,  "family": "mixed"},
    {"name": "phi:latest",          "label": "Phi\n2.7B",         "params_b": 2.7,  "family": "mixed"},
    {"name": "mistral:latest",      "label": "Mistral\n7B",       "params_b": 7.0,  "family": "mixed"},
    # Qwen3.5 family (same architecture, different sizes)
    {"name": "qwen3.5:0.8b",        "label": "Qwen3.5\n0.8B",    "params_b": 0.8,  "family": "qwen3.5"},
    {"name": "qwen3.5:2b",          "label": "Qwen3.5\n2B",      "params_b": 2.0,  "family": "qwen3.5"},
    {"name": "qwen3.5:4b",          "label": "Qwen3.5\n4B",      "params_b": 4.0,  "family": "qwen3.5"},
    {"name": "qwen3.5:9b",          "label": "Qwen3.5\n9B",      "params_b": 9.0,  "family": "qwen3.5"},
    {"name": "qwen3.5:397b-cloud",  "label": "Qwen3.5\n397B",    "params_b": 397.0,"family": "qwen3.5"},
]

# ============================================================
# TASKS — flexible keyword groups
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
# MEMORY STORE — persistent per model
# Saves to disk so memory survives between runs
# ============================================================

memory_store = []

def get_memory_path(model_name):
    safe = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
    return os.path.join(MEMORY_DIR, f"memory_{safe}.json")

def load_memory(model_name):
    global memory_store
    path = get_memory_path(model_name)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        memory_store = [
            {"task": e["task"], "plan": e["plan"],
             "keywords": set(e["keywords"])}
            for e in data
        ]
        print(f"  [MEMORY] Loaded {len(memory_store)} entries from {path}")
    else:
        memory_store = []

def save_memory(model_name):
    path = get_memory_path(model_name)
    data = [
        {"task": e["task"], "plan": e["plan"],
         "keywords": list(e["keywords"])}
        for e in memory_store
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

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

def get_timeout(model_name):
    if "cloud" in model_name:   return 120
    elif "9b"   in model_name:  return 180
    elif "4b"   in model_name:  return 120
    elif "7b"   in model_name.lower() or "mistral" in model_name: return 120
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
    # Only qwen3.5 supports think=False
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
# PROMPTS
# ============================================================

def prompt_no_memory(task):
    return (
        "You are a battery verification engineer.\n"
        "Generate a numbered step-by-step technical plan.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\nPLAN:\n1."
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
        "Using the memory above as guidance, generate a numbered step-by-step "
        "technical plan for the task below.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\nPLAN:\n1."
    )

# ============================================================
# OUTPUT CLEANING
# ============================================================

def extract_plan(text):
    if not text:
        return ""
    skip = [
        "analyze the request", "determine the plan", "determine the content",
        "drafting", "evaluate constraints", "constraint check", "thinking process",
        "numbered steps only", "max 6 steps", "no extra text", "wait,",
        "actually,", "let me", "refine steps", "refining", "analyze the task",
        "analyze the memory", "review memory", "finalizing",
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

# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

def avg(lst):
    return round(sum(lst) / len(lst), 3) if lst else 0.0

results = {}

for model_info in MODELS:
    model_name = model_info["name"]

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}  [{model_info['family']}]")
    print("=" * 60)

    # Fresh memory for each model
    memory_store.clear()

    model_results = {k: [] for k in [
        "coverage_no_mem", "coverage_mem",
        "latency_no_mem",  "latency_mem",
        "tokens_no_mem",   "tokens_mem",
    ]}

    for t in TASKS:
        task     = t["task"]
        keywords = t["expected_keywords"]

        print(f"\n  TASK: {task}")

        # WITHOUT MEMORY
        raw1, lat1, tok1 = generate(model_name, prompt_no_memory(task))
        plan1 = extract_plan(raw1)
        sc1   = score_plan(plan1, keywords)
        print(f"\n  --- WITHOUT MEMORY ---")
        for line in plan1.split("\n")[:5]:
            print(f"    {line}")
        print(f"  → coverage={sc1}  latency={lat1}s  tokens={tok1}")

        # WITH MEMORY
        raw2, lat2, tok2 = generate(model_name, prompt_with_memory(task))
        plan2 = extract_plan(raw2)
        sc2   = score_plan(plan2, keywords)
        print(f"\n  --- WITH MEMORY ---")
        for line in plan2.split("\n")[:5]:
            print(f"    {line}")
        print(f"  → coverage={sc2}  latency={lat2}s  tokens={tok2}")

        add_to_memory(task, plan2)

        model_results["coverage_no_mem"].append(sc1)
        model_results["coverage_mem"].append(sc2)
        model_results["latency_no_mem"].append(lat1)
        model_results["latency_mem"].append(lat2)
        model_results["tokens_no_mem"].append(tok1)
        model_results["tokens_mem"].append(tok2)

    # Save memory to disk
    save_memory(model_name)

    results[model_name] = model_results
    print(f"\n  ── KPI SUMMARY ──")
    print(f"  Step coverage: {avg(model_results['coverage_no_mem'])} → {avg(model_results['coverage_mem'])}")
    print(f"  Latency (avg): {avg(model_results['latency_no_mem'])}s → {avg(model_results['latency_mem'])}s")
    print(f"  Tokens (avg):  {avg(model_results['tokens_no_mem'])} → {avg(model_results['tokens_mem'])}")

# ============================================================
# PLOTTING
# ============================================================

model_names  = [m["name"]   for m in MODELS]
model_labels = [m["label"]  for m in MODELS]
families     = [m["family"] for m in MODELS]

# Colors per family
C_MIXED = "#888780"
C_QWEN  = "#185FA5"
C_MEM_MIXED = "#444441"
C_MEM_QWEN  = "#0C447C"

bar_colors_base = [C_MIXED if f == "mixed" else C_QWEN for f in families]
bar_colors_mem  = [C_MEM_MIXED if f == "mixed" else C_MEM_QWEN for f in families]

x = np.arange(len(MODELS))
w = 0.35

# Plot 1: Step coverage
fig, ax = plt.subplots(figsize=(14, 5))
cov_no  = [avg(results[m]["coverage_no_mem"]) for m in model_names]
cov_mem = [avg(results[m]["coverage_mem"])    for m in model_names]

bars1 = ax.bar(x - w/2, cov_no,  w, color=bar_colors_base, alpha=0.85, label="Baseline (no memory)")
bars2 = ax.bar(x + w/2, cov_mem, w, color=bar_colors_mem,  alpha=0.90, label="Optimized (with memory)")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=9, color="#444441")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=9, color="#0C447C")

for i, (v1, v2) in enumerate(zip(cov_no, cov_mem)):
    delta = v2 - v1
    col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
    ax.text(x[i], max(v1, v2) + 0.07, f"{delta:+.2f}",
            ha="center", fontsize=9, color=col, fontweight="bold")

# Family divider line
ax.axvline(x=2.5, color="#B4B2A9", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(1.0,  1.08, "Mixed Families", ha="center", fontsize=9, color=C_MIXED, style="italic")
ax.text(5.5,  1.08, "Qwen3.5 Family", ha="center", fontsize=9, color=C_QWEN,  style="italic")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Step Coverage Score (0-1)", fontsize=11)
ax.set_title("Experiment 1 — Architecture & Size Comparison: Step Coverage",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.18)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "exp1_step_coverage.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved: {path1}")

# Plot 2: Latency
fig, ax = plt.subplots(figsize=(14, 5))
lat_no  = [avg(results[m]["latency_no_mem"]) for m in model_names]
lat_mem = [avg(results[m]["latency_mem"])    for m in model_names]

ax.bar(x - w/2, lat_no,  w, color=bar_colors_base, alpha=0.85, label="Baseline")
ax.bar(x + w/2, lat_mem, w, color=bar_colors_mem,  alpha=0.90, label="With memory")

for i, (v1, v2) in enumerate(zip(lat_no, lat_mem)):
    ax.text(x[i] - w/2, v1 + 0.3, f"{v1:.1f}s", ha="center", fontsize=8, color="#444441")
    ax.text(x[i] + w/2, v2 + 0.3, f"{v2:.1f}s", ha="center", fontsize=8, color="#0C447C")

ax.axvline(x=2.5, color="#B4B2A9", linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title("Experiment 1 — Latency Comparison: Baseline vs Memory Optimized",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "exp1_latency.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path2}")

# Plot 3: Token count
fig, ax = plt.subplots(figsize=(14, 5))
tok_no  = [avg(results[m]["tokens_no_mem"]) for m in model_names]
tok_mem = [avg(results[m]["tokens_mem"])    for m in model_names]

ax.bar(x - w/2, tok_no,  w, color=bar_colors_base, alpha=0.85, label="Baseline")
ax.bar(x + w/2, tok_mem, w, color=bar_colors_mem,  alpha=0.90, label="With memory")

for i, (v1, v2) in enumerate(zip(tok_no, tok_mem)):
    ax.text(x[i] - w/2, v1 + 1, f"{int(v1)}", ha="center", fontsize=8, color="#444441")
    ax.text(x[i] + w/2, v2 + 1, f"{int(v2)}", ha="center", fontsize=8, color="#0C447C")

ax.axvline(x=2.5, color="#B4B2A9", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(1.0, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 10,
        "Mixed Families", ha="center", fontsize=9, color=C_MIXED, style="italic")
ax.text(5.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 10,
        "Qwen3.5 Family", ha="center", fontsize=9, color=C_QWEN, style="italic")
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Average Token Count", fontsize=11)
ax.set_title("Experiment 1 — Token Count: Baseline vs Memory Optimized",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "exp1_tokens.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path3}")
serializable = {
    m: {k: [float(v) for v in vals] for k, vals in r.items()}
    for m, r in results.items()
}
path_json = os.path.join(PLOTS_DIR, "exp1_results.json")
with open(path_json, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"✅ Saved: {path_json}")

print("\n" + "=" * 60)
print("EXPERIMENT 1 COMPLETE")
print("=" * 60)