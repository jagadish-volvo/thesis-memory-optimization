"""
Experiment 3 — Semantic Memory (MemPalace-Inspired)
=====================================================
Compares two conditions across all 8 models:
  1. BASELINE     — model solves task alone, no memory
  2. SEMANTIC MEM — model uses ChromaDB semantic memory
                    (nomic-embed-text embeddings, cosine similarity)

Inspired by the MemPalace approach:
  - Store task solutions as vector embeddings in ChromaDB
  - Retrieve semantically similar past solutions using cosine similarity
  - Inject retrieved solutions as context before generating new plan

Key difference from keyword memory (Experiment 1):
  - Keyword memory: exact word overlap matching
  - Semantic memory: meaning-based similarity via embeddings
    (finds relevant memories even when different words are used)

KPIs: Step Coverage (deterministic), Latency (measured), Token Count
"""

import requests
import time
import json
import os
import re
import chromadb
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL   = "http://localhost:11434/api/chat"
EMBED_URL    = "http://localhost:11434/api/embeddings"
EMBED_MODEL  = "nomic-embed-text:latest"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots_exp3")
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}\n")

# All 8 models — same as Experiment 1
MODELS = [
    {"name": "tinyllama:1.1b",     "label": "TinyLlama\n1.1B",  "params_b": 1.1,   "family": "mixed"},
    {"name": "phi:latest",         "label": "Phi\n2.7B",         "params_b": 2.7,   "family": "mixed"},
    {"name": "mistral:latest",     "label": "Mistral\n7B",       "params_b": 7.0,   "family": "mixed"},
    {"name": "qwen3.5:0.8b",       "label": "Qwen3.5\n0.8B",    "params_b": 0.8,   "family": "qwen3.5"},
    {"name": "qwen3.5:2b",         "label": "Qwen3.5\n2B",      "params_b": 2.0,   "family": "qwen3.5"},
    {"name": "qwen3.5:4b",         "label": "Qwen3.5\n4B",      "params_b": 4.0,   "family": "qwen3.5"},
    {"name": "qwen3.5:9b",         "label": "Qwen3.5\n9B",      "params_b": 9.0,   "family": "qwen3.5"},
    {"name": "qwen3.5:397b-cloud", "label": "Qwen3.5\n397B",    "params_b": 397.0, "family": "qwen3.5"},
]

# ============================================================
# TASKS — same as Experiments 1 and 2
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
# SEMANTIC MEMORY — ChromaDB + nomic-embed-text
# MemPalace-inspired: store everything, retrieve by meaning
# ============================================================

# Global ChromaDB client — in-memory, resets per model (fair comparison)
chroma_client = None
chroma_collection = None

def init_semantic_memory(model_name):
    """
    Initialise a fresh ChromaDB in-memory collection for this model.
    Fresh per model ensures fair comparison — same as Experiment 1.
    """
    global chroma_client, chroma_collection
    # EphemeralClient = pure in-memory, no disk persistence needed
    chroma_client = chromadb.EphemeralClient()
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    collection_name = f"memory_{safe_name}"
    chroma_collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for semantic search
    )
    print(f"  [MEMORY] Initialised ChromaDB collection: {collection_name}")

def get_embedding(text):
    """
    Get embedding vector from nomic-embed-text via Ollama.
    nomic-embed-text is a local embedding model — no API key, no cloud.
    """
    try:
        r = requests.post(EMBED_URL, json={
            "model": EMBED_MODEL,
            "prompt": text
        }, timeout=30)
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as e:
        print(f"  [EMBED ERROR] {e}")
        return []

def add_to_semantic_memory(task, plan_text):
    """
    Store a solved task in ChromaDB using nomic-embed-text embeddings.
    The embedding is computed from the task + plan combined for richer retrieval.
    """
    if chroma_collection is None:
        return
    combined_text = f"Task: {task}\nSolution: {plan_text}"
    embedding = get_embedding(combined_text)
    if not embedding:
        return
    doc_id = f"task_{chroma_collection.count()}"
    chroma_collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[combined_text],
        metadatas=[{"task": task, "plan": plan_text}]
    )

def retrieve_from_semantic_memory(task, top_k=2):
    """
    Retrieve the most semantically similar past solutions using cosine similarity.
    This is the key difference from keyword memory — finds relevant memories
    based on MEANING, not exact word overlap.
    """
    if chroma_collection is None or chroma_collection.count() == 0:
        return []
    query_embedding = get_embedding(task)
    if not query_embedding:
        return []
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, chroma_collection.count()),
        include=["metadatas", "distances"]
    )
    entries = []
    if results and results["metadatas"]:
        for meta in results["metadatas"][0]:
            entries.append({
                "task": meta.get("task", ""),
                "plan": meta.get("plan", "")
            })
    return entries

# ============================================================
# GENERATION
# ============================================================

def get_timeout(model_name):
    if "cloud"    in model_name: return 120
    elif "9b"     in model_name: return 180
    elif "4b"     in model_name: return 120
    elif "mistral" in model_name: return 120
    else:                        return 90

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
# PROMPTS
# ============================================================

def prompt_baseline(task):
    return (
        "You are a battery verification engineer.\n"
        "Generate a numbered step-by-step technical plan.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\nPLAN:\n1."
    )

def prompt_semantic_memory(task):
    relevant = retrieve_from_semantic_memory(task)
    if not relevant:
        return prompt_baseline(task)

    mem_lines = ["[SEMANTIC MEMORY — semantically similar past solutions]"]
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
        "Using the semantically similar past solutions above as guidance, "
        "generate a numbered step-by-step technical plan for the task below.\n"
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
# MAIN EXPERIMENT LOOP
# ============================================================

results = {}

for model_info in MODELS:
    model_name = model_info["name"]

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}  [{model_info['family']}]")
    print("=" * 60)

    # Fresh ChromaDB collection per model — fair comparison
    init_semantic_memory(model_name)

    model_results = {k: [] for k in [
        "coverage_baseline", "coverage_semantic",
        "latency_baseline",  "latency_semantic",
        "tokens_baseline",   "tokens_semantic",
    ]}

    for t in TASKS:
        task     = t["task"]
        keywords = t["expected_keywords"]

        print(f"\n  TASK: {task}")

        # ---- BASELINE — no memory ----
        raw1, lat1, tok1 = generate(model_name, prompt_baseline(task))
        plan1 = extract_plan(raw1)
        sc1   = score_plan(plan1, keywords)

        print(f"\n  --- BASELINE (no memory) ---")
        for line in plan1.split("\n")[:5]:
            print(f"    {line}")
        print(f"  → coverage={sc1}  latency={lat1}s  tokens={tok1}")

        # ---- SEMANTIC MEMORY ----
        raw2, lat2, tok2 = generate(model_name, prompt_semantic_memory(task))
        plan2 = extract_plan(raw2)
        sc2   = score_plan(plan2, keywords)

        print(f"\n  --- WITH SEMANTIC MEMORY ---")
        for line in plan2.split("\n")[:5]:
            print(f"    {line}")
        print(f"  → coverage={sc2}  latency={lat2}s  tokens={tok2}")

        # Store plan in semantic memory for future tasks
        add_to_semantic_memory(task, plan2)

        model_results["coverage_baseline"].append(sc1)
        model_results["coverage_semantic"].append(sc2)
        model_results["latency_baseline"].append(lat1)
        model_results["latency_semantic"].append(lat2)
        model_results["tokens_baseline"].append(tok1)
        model_results["tokens_semantic"].append(tok2)

    results[model_name] = model_results
    print(f"\n  ── KPI SUMMARY ──")
    print(f"  Step coverage: {avg(model_results['coverage_baseline'])} → {avg(model_results['coverage_semantic'])}")
    print(f"  Latency (avg): {avg(model_results['latency_baseline'])}s → {avg(model_results['latency_semantic'])}s")
    print(f"  Tokens (avg):  {avg(model_results['tokens_baseline'])} → {avg(model_results['tokens_semantic'])}")

# ============================================================
# PLOTTING
# ============================================================

model_names  = [m["name"]  for m in MODELS]
model_labels = [m["label"] for m in MODELS]
families     = [m["family"] for m in MODELS]

C_BASE        = "#888780"
C_SEM         = "#185FA5"
C_BASE_DARK   = "#444441"
C_SEM_DARK    = "#0C447C"
C_MIXED_LINE  = "#B4B2A9"

bar_colors_base = [C_BASE  for _ in families]
bar_colors_sem  = [C_SEM   for _ in families]

x = np.arange(len(MODELS))
w = 0.35

# ── Plot 1: Step coverage ──
fig, ax = plt.subplots(figsize=(14, 5))

cov_base = [avg(results[m]["coverage_baseline"]) for m in model_names]
cov_sem  = [avg(results[m]["coverage_semantic"])  for m in model_names]

bars1 = ax.bar(x - w/2, cov_base, w, color=C_BASE, alpha=0.85, label="Baseline (no memory)")
bars2 = ax.bar(x + w/2, cov_sem,  w, color=C_SEM,  alpha=0.90, label="Semantic memory (ChromaDB)")

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=9, color=C_BASE_DARK)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=9, color=C_SEM_DARK)

for i, (v1, v2) in enumerate(zip(cov_base, cov_sem)):
    delta = v2 - v1
    col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
    ax.text(x[i], max(v1, v2) + 0.07, f"{delta:+.2f}",
            ha="center", fontsize=9, color=col, fontweight="bold")

# Family divider
ax.axvline(x=2.5, color=C_MIXED_LINE, linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(1.0, 1.10, "Mixed Families", ha="center", fontsize=9, color="#888780", style="italic")
ax.text(5.5, 1.10, "Qwen3.5 Family", ha="center", fontsize=9, color=C_SEM_DARK, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Step Coverage Score (0-1)", fontsize=11)
ax.set_title("Experiment 3 — Semantic Memory (ChromaDB): Step Coverage Across All Models",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.20)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path1 = os.path.join(PLOTS_DIR, "exp3_step_coverage.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Saved: {path1}")

# ── Plot 2: Latency ──
fig, ax = plt.subplots(figsize=(14, 5))

lat_base = [avg(results[m]["latency_baseline"]) for m in model_names]
lat_sem  = [avg(results[m]["latency_semantic"])  for m in model_names]

ax.bar(x - w/2, lat_base, w, color=C_BASE, alpha=0.85, label="Baseline")
ax.bar(x + w/2, lat_sem,  w, color=C_SEM,  alpha=0.90, label="Semantic memory")

for i, (v1, v2) in enumerate(zip(lat_base, lat_sem)):
    ax.text(x[i] - w/2, v1 + 0.3, f"{v1:.1f}s", ha="center", fontsize=8, color=C_BASE_DARK)
    ax.text(x[i] + w/2, v2 + 0.3, f"{v2:.1f}s", ha="center", fontsize=8, color=C_SEM_DARK)

ax.axvline(x=2.5, color=C_MIXED_LINE, linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Average Latency (seconds)", fontsize=11)
ax.set_title("Experiment 3 — Latency: Baseline vs Semantic Memory",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path2 = os.path.join(PLOTS_DIR, "exp3_latency.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path2}")

# ── Plot 3: Token count ──
fig, ax = plt.subplots(figsize=(14, 5))

tok_base = [avg(results[m]["tokens_baseline"]) for m in model_names]
tok_sem  = [avg(results[m]["tokens_semantic"])  for m in model_names]

ax.bar(x - w/2, tok_base, w, color=C_BASE, alpha=0.85, label="Baseline")
ax.bar(x + w/2, tok_sem,  w, color=C_SEM,  alpha=0.90, label="Semantic memory")

for i, (v1, v2) in enumerate(zip(tok_base, tok_sem)):
    ax.text(x[i] - w/2, v1 + 1, f"{int(v1)}", ha="center", fontsize=8, color=C_BASE_DARK)
    ax.text(x[i] + w/2, v2 + 1, f"{int(v2)}", ha="center", fontsize=8, color=C_SEM_DARK)

ax.axvline(x=2.5, color=C_MIXED_LINE, linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Average Token Count", fontsize=11)
ax.set_title("Experiment 3 — Token Count: Baseline vs Semantic Memory",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path3 = os.path.join(PLOTS_DIR, "exp3_tokens.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path3}")

# ── Plot 4: Coverage improvement bar ──
fig, ax = plt.subplots(figsize=(12, 4))
improvements = [avg(results[m]["coverage_semantic"]) - avg(results[m]["coverage_baseline"])
                for m in model_names]
colors = ["#0F6E56" if v > 0 else ("#A32D2D" if v < 0 else "#888780")
          for v in improvements]
bars = ax.bar(x, improvements, 0.5, color=colors, alpha=0.85)
for bar, val in zip(bars, improvements):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.005 if val >= 0 else -0.02),
            f"{val:+.2f}", ha="center", fontsize=10,
            fontweight="bold", color="#444441")
ax.axhline(y=0, color="#444441", linewidth=0.8)
ax.axvline(x=2.5, color=C_MIXED_LINE, linestyle="--", linewidth=1.5, alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.set_ylabel("Coverage Improvement", fontsize=11)
ax.set_title("Experiment 3 — Coverage Improvement from Semantic Memory",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
path4 = os.path.join(PLOTS_DIR, "exp3_improvement.png")
plt.savefig(path4, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path4}")

# ── Save JSON ──
serializable = {
    m: {k: [float(v) for v in vals] for k, vals in r.items()}
    for m, r in results.items()
}
path_json = os.path.join(PLOTS_DIR, "exp3_results.json")
with open(path_json, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"✅ Saved: {path_json}")

print("\n" + "=" * 60)
print("EXPERIMENT 3 COMPLETE")
print("=" * 60)
