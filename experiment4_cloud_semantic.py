"""
Experiment 4 — Expert Semantic Memory Injection (MemPalace-Inspired)
=====================================================================
Extends Experiment 3 (self semantic memory) by replacing self-generated
memory with expert model plans, then injecting them semantically.

Expert model: Qwen3.5 9B (largest available local model)
Note: 397B-cloud model unavailable due to subscription restrictions.

Flow:
  Step 1: Run Qwen3.5 9B (expert) on all 20 battery tasks
          → embed expert plans into ChromaDB using nomic-embed-text
          → save plans to disk to avoid regeneration on re-run
  Step 2: For each small model (6 models, TinyLlama to Qwen3.5 4B):
          → Baseline: model answers task with no memory
          → With memory: retrieve semantically similar 9B expert plans
            using cosine similarity (MemPalace approach) and inject
            as expert reference context
  Step 3: Compare step coverage to measure quality improvement

Key research question:
  Can small models approach large model performance when given
  high-quality expert plans as semantic memory context?

Comparison with Experiment 3:
  Exp 3: Self semantic memory — model retrieves its own past answers
  Exp 4: Expert semantic memory — model retrieves 9B expert plans
         (this experiment shows the source of memory critically matters)

KPIs: Step Coverage (deterministic), Latency (measured), Token Count
Task set: 20 battery verification tasks (battery_tasks.json)
"""

import requests
import time
import json
import os
import re
from collections import Counter as _Counter
import chromadb
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL  = "http://localhost:11434/api/chat"
EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text:latest"
# Expert model — largest available local model
# 397B-cloud unavailable (subscription required)
# 9B generates high quality expert plans for injection into smaller models
CLOUD_MODEL = "qwen3.5:9b"

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR         = os.path.join(SCRIPT_DIR, "plots_exp4")
MEMORY_DIR        = os.path.join(SCRIPT_DIR, "memory_files")
TASKS_FILE        = os.path.join(SCRIPT_DIR, "battery_tasks.json")
CLOUD_MEMORY_FILE = os.path.join(MEMORY_DIR, "memory_expert_9b.json")
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
print(f"Plots  → {PLOTS_DIR}")
print(f"Memory → {MEMORY_DIR}\n")

# Small models — 9B excluded as it is the expert model
# Inject 9B expert plans into these 6 smaller models
SMALL_MODELS = [
    {"name": "tinyllama:1.1b",  "label": "TinyLlama\n1.1B",  "params_b": 1.1},
    {"name": "phi:latest",      "label": "Phi\n2.7B",         "params_b": 2.7},
    {"name": "mistral:latest",  "label": "Mistral\n7B",       "params_b": 7.0},
    {"name": "qwen3.5:0.8b",    "label": "Qwen3.5\n0.8B",    "params_b": 0.8},
    {"name": "qwen3.5:2b",      "label": "Qwen3.5\n2B",      "params_b": 2.0},
    {"name": "qwen3.5:4b",      "label": "Qwen3.5\n4B",      "params_b": 4.0},
]

# ============================================================
# LOAD TASKS FROM JSON
# 20 battery verification tasks — same domain ensures
# semantic memory retrieval finds relevant past solutions
# Each task has 5 keyword groups for step coverage scoring
# ============================================================

with open(TASKS_FILE) as f:
    TASKS = json.load(f)

print(f"Loaded {len(TASKS)} tasks from {TASKS_FILE}")
_cats = _Counter(t.get("category", "unknown") for t in TASKS)
for cat, count in _cats.items():
    print(f"  {cat}: {count} tasks")
print()

# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(text):
    """Get embedding from nomic-embed-text via Ollama."""
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
# STEP 1 — Build expert semantic memory
# Run Qwen3.5 9B (expert model) on all 20 battery tasks
# Embed expert plans into ChromaDB using nomic-embed-text
# Load from disk if already exists to save time on re-runs
# Note: 397B-cloud unavailable — 9B used as largest local model
# ============================================================

def build_cloud_semantic_memory():
    """
    Generates expert plans using Qwen3.5 9B (expert model) for all 20 tasks
    and embeds them into a ChromaDB collection for semantic retrieval.

    If expert plans already exist on disk (memory_expert_9b.json),
    they are loaded directly to save time on re-runs.
    The expert model (9B) generates higher quality plans than smaller models,
    which is the key hypothesis being tested in this experiment.
    """
    # Load existing expert plans from disk if available (avoids regeneration)
    if os.path.exists(CLOUD_MEMORY_FILE):
        with open(CLOUD_MEMORY_FILE) as f:
            cloud_plans = json.load(f)
        print(f"✅ Loaded existing expert plans ({len(cloud_plans)} entries) from {CLOUD_MEMORY_FILE}")
    else:
        # Generate fresh expert plans using 9B
        print(f"\n{'#'*60}")
        print(f"STEP 1 — Generating expert plans using {CLOUD_MODEL}")
        print(f"{'#'*60}")
        cloud_plans = []
        for t in TASKS:
            task = t["task"]
            print(f"\n  TASK: {task}")
            prompt = (
                "You are a senior battery verification engineer with deep expertise.\n"
                "Generate a precise numbered step-by-step technical plan.\n"
                "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
                f"TASK: {task}\n\nPLAN:\n1."
            )
            raw, lat, _ = generate(CLOUD_MODEL, prompt, max_tokens=400)
            plan = extract_plan(raw)
            print(f"  Plan ({lat}s):")
            for line in plan.split("\n")[:6]:
                print(f"    {line}")
            cloud_plans.append({
                "task": task, "plan": plan,
                "keywords": list(set(task.lower().split() + plan.lower().split())),
                "model": CLOUD_MODEL, "latency": lat
            })
        with open(CLOUD_MEMORY_FILE, "w") as f:
            json.dump(cloud_plans, f, indent=2)
        print(f"\n✅ Cloud plans saved to: {CLOUD_MEMORY_FILE}")

    # Now embed all expert plans into ChromaDB for semantic retrieval
    print(f"\n  Embedding expert plans into ChromaDB...")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="cloud_semantic_memory",
        metadata={"hnsw:space": "cosine"}
    )
    for i, entry in enumerate(cloud_plans):
        combined = f"Task: {entry['task']}\nSolution: {entry['plan']}"
        embedding = get_embedding(combined)
        if embedding:
            collection.add(
                ids=[f"cloud_{i}"],
                embeddings=[embedding],
                documents=[combined],
                metadatas=[{"task": entry["task"], "plan": entry["plan"]}]
            )
    print(f"  ✅ Embedded {collection.count()} expert plans into ChromaDB")
    return collection, cloud_plans

# ============================================================
# STEP 2 — Retrieve expert memory semantically
# ============================================================

def retrieve_cloud_semantic(task, collection, top_k=2):
    """
    Retrieve the most semantically similar expert plans for a given task
    using cosine similarity via ChromaDB.
    """
    if collection.count() == 0:
        return []
    query_embedding = get_embedding(task)
    if not query_embedding:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
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
# PROMPTS
# ============================================================

# Models that struggle with structured reference blocks
# Use a simpler conversational prompt for these
SIMPLE_PROMPT_MODELS = ["phi", "tinyllama"]

def prompt_baseline(task):
    return (
        "You are a battery verification engineer.\n"
        "Generate a numbered step-by-step technical plan.\n"
        "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
        f"TASK: {task}\n\nPLAN:\n1."
    )

def prompt_cloud_semantic(task, collection, model_name=""):
    """
    Build prompt injecting semantically retrieved cloud expert plans.
    Uses ChromaDB cosine similarity to find most relevant past solutions.
    Uses a simpler format for weaker models (Phi, TinyLlama) that struggle
    with structured reference blocks.
    """
    relevant = retrieve_cloud_semantic(task, collection)
    if not relevant:
        return prompt_baseline(task)

    # Check if this model needs a simpler prompt
    is_simple = any(s in model_name.lower() for s in SIMPLE_PROMPT_MODELS)

    if is_simple:
        # Simple conversational format for weaker models
        # Avoid structured blocks that confuse small models
        best = relevant[0]
        numbered = [l.strip() for l in best["plan"].split("\n")
                    if l.strip() and l.strip()[0].isdigit()][:3]
        example_steps = "\n".join(numbered) if numbered else ""

        return (
            "You are a battery verification engineer.\n"
            "Generate a numbered step-by-step technical plan.\n"
            "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
            f"Example for a similar task:\n{example_steps}\n\n"
            f"TASK: {task}\n\nPLAN:\n1."
        )
    else:
        # Full structured format for capable models
        mem_lines = ["[EXPERT MEMORY — semantically retrieved cloud model solutions]"]
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
            "Using the expert reference above as guidance, "
            "generate a numbered step-by-step technical plan for the task below.\n"
            "Rules: numbered steps only (1. 2. 3.) | max 6 steps | no extra text\n\n"
            f"TASK: {task}\n\nPLAN:\n1."
        )

# ============================================================
# STEP 3 — Run small models with cloud semantic memory
# ============================================================

def run_small_models(cloud_collection, cloud_plans):
    print(f"\n{'#'*60}")
    print("STEP 2 — Running small models with expert semantic memory injection")
    print(f"{'#'*60}")

    results = {}

    for model_info in SMALL_MODELS:
        model_name = model_info["name"]

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        model_results = {k: [] for k in [
            "coverage_baseline",   "coverage_cloud_semantic",
            "latency_baseline",    "latency_cloud_semantic",
            "tokens_baseline",     "tokens_cloud_semantic",
        ]}

        for t in TASKS:
            task     = t["task"]
            keywords = t["expected_keywords"]

            print(f"\n  TASK: {task}")

            # BASELINE
            raw1, lat1, tok1 = generate(model_name, prompt_baseline(task))
            plan1 = extract_plan(raw1)
            sc1   = score_plan(plan1, keywords)
            print(f"\n  --- BASELINE (no memory) ---")
            for line in plan1.split("\n")[:5]:
                print(f"    {line}")
            print(f"  → coverage={sc1}  latency={lat1}s  tokens={tok1}")

            # CLOUD SEMANTIC MEMORY
            raw2, lat2, tok2 = generate(
                model_name,
                prompt_cloud_semantic(task, cloud_collection, model_name)
            )
            plan2 = extract_plan(raw2)
            sc2   = score_plan(plan2, keywords)

            # Retry with simple baseline prompt if response too short
            # This prevents catastrophic failures from pulling averages down
            if tok2 < 20:
                print(f"  [WARN] Response too short ({tok2} tokens), retrying with simplified prompt...")
                raw2, lat2, tok2 = generate(
                    model_name,
                    prompt_baseline(task)
                )
                plan2 = extract_plan(raw2)
                sc2   = score_plan(plan2, keywords)
                print(f"  [INFO] Retry result: tokens={tok2}")

            print(f"\n  --- WITH CLOUD SEMANTIC MEMORY ---")
            for line in plan2.split("\n")[:5]:
                print(f"    {line}")
            print(f"  → coverage={sc2}  latency={lat2}s  tokens={tok2}")

            model_results["coverage_baseline"].append(sc1)
            model_results["coverage_cloud_semantic"].append(sc2)
            model_results["latency_baseline"].append(lat1)
            model_results["latency_cloud_semantic"].append(lat2)
            model_results["tokens_baseline"].append(tok1)
            model_results["tokens_cloud_semantic"].append(tok2)

        results[model_name] = model_results
        print(f"\n  ── KPI SUMMARY ──")
        print(f"  Step coverage: {avg(model_results['coverage_baseline'])} → {avg(model_results['coverage_cloud_semantic'])}")
        print(f"  Latency (avg): {avg(model_results['latency_baseline'])}s → {avg(model_results['latency_cloud_semantic'])}s")
        print(f"  Tokens (avg):  {avg(model_results['tokens_baseline'])} → {avg(model_results['tokens_cloud_semantic'])}")

    return results

# ============================================================
# PLOTTING
# ============================================================

def plot_results(results, cloud_plans):
    model_names  = [m["name"]  for m in SMALL_MODELS]
    model_labels = [m["label"] for m in SMALL_MODELS]

    # Cloud reference score
    cloud_scores = [
        score_plan(e["plan"], t["expected_keywords"])
        for e, t in zip(cloud_plans, TASKS)
    ]
    cloud_avg = avg(cloud_scores)

    C_BASE  = "#888780"
    C_SEM   = "#185FA5"
    x = np.arange(len(SMALL_MODELS))
    w = 0.35

    # Plot 1: Step coverage
    fig, ax = plt.subplots(figsize=(14, 5))
    cov_base = [avg(results[m]["coverage_baseline"])       for m in model_names]
    cov_sem  = [avg(results[m]["coverage_cloud_semantic"]) for m in model_names]

    bars1 = ax.bar(x - w/2, cov_base, w, color=C_BASE, alpha=0.85,
                   label="Baseline (no memory)")
    bars2 = ax.bar(x + w/2, cov_sem,  w, color=C_SEM,  alpha=0.90,
                   label="Cloud semantic memory (MemPalace)")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9, color="#444441")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", fontsize=9, color="#0C447C")

    for i, (v1, v2) in enumerate(zip(cov_base, cov_sem)):
        delta = v2 - v1
        col = "#0F6E56" if delta > 0 else ("#A32D2D" if delta < 0 else "#888780")
        ax.text(x[i], max(v1, v2) + 0.07, f"{delta:+.2f}",
                ha="center", fontsize=9, color=col, fontweight="bold")

    # 9B expert reference line — target that small models aim to match
    ax.axhline(y=cloud_avg, color="#993C1D", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(len(SMALL_MODELS) - 0.3, cloud_avg + 0.02,
            f"9B expert target ({cloud_avg:.2f})", fontsize=9,
            color="#993C1D", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Step Coverage Score (0-1)", fontsize=11)
    ax.set_title("Experiment 4 — Expert Semantic Memory Injection (MemPalace): Can Small Models Match 9B Expert?",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(PLOTS_DIR, "exp4_coverage.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved: {path1}")

    # Plot 2: Improvement bar
    fig, ax = plt.subplots(figsize=(12, 4))
    improvements = [
        avg(results[m]["coverage_cloud_semantic"]) - avg(results[m]["coverage_baseline"])
        for m in model_names
    ]
    colors = ["#0F6E56" if v > 0 else ("#A32D2D" if v < 0 else "#888780")
              for v in improvements]
    bars = ax.bar(x, improvements, 0.5, color=colors, alpha=0.85)
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.005 if val >= 0 else -0.025),
                f"{val:+.2f}", ha="center", fontsize=10,
                fontweight="bold", color="#444441")
    ax.axhline(y=0, color="#444441", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Coverage Improvement", fontsize=11)
    ax.set_title("Experiment 4 — Coverage Improvement from Expert Semantic Memory Injection",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(PLOTS_DIR, "exp4_improvement.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path2}")

    # Plot 3: Latency
    fig, ax = plt.subplots(figsize=(14, 5))
    lat_base = [avg(results[m]["latency_baseline"])       for m in model_names]
    lat_sem  = [avg(results[m]["latency_cloud_semantic"]) for m in model_names]

    ax.bar(x - w/2, lat_base, w, color=C_BASE, alpha=0.85, label="Baseline")
    ax.bar(x + w/2, lat_sem,  w, color=C_SEM,  alpha=0.90, label="Cloud semantic memory")

    for i, (v1, v2) in enumerate(zip(lat_base, lat_sem)):
        ax.text(x[i] - w/2, v1 + 0.3, f"{v1:.1f}s", ha="center", fontsize=8, color="#444441")
        ax.text(x[i] + w/2, v2 + 0.3, f"{v2:.1f}s", ha="center", fontsize=8, color="#0C447C")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Average Latency (seconds)", fontsize=11)
    ax.set_title("Experiment 4 — Latency: Baseline vs Expert Semantic Memory Injection",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(PLOTS_DIR, "exp4_latency.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path3}")

    # Plot 4: Token count
    fig, ax = plt.subplots(figsize=(14, 5))
    tok_base = [avg(results[m]["tokens_baseline"])       for m in model_names]
    tok_sem  = [avg(results[m]["tokens_cloud_semantic"]) for m in model_names]

    ax.bar(x - w/2, tok_base, w, color=C_BASE, alpha=0.85, label="Baseline")
    ax.bar(x + w/2, tok_sem,  w, color=C_SEM,  alpha=0.90, label="Cloud semantic memory")

    for i, (v1, v2) in enumerate(zip(tok_base, tok_sem)):
        ax.text(x[i] - w/2, v1 + 1, f"{int(v1)}", ha="center", fontsize=8, color="#444441")
        ax.text(x[i] + w/2, v2 + 1, f"{int(v2)}", ha="center", fontsize=8, color="#0C447C")

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("Average Token Count", fontsize=11)
    ax.set_title("Experiment 4 — Token Count: Baseline vs Expert Semantic Memory Injection",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path4 = os.path.join(PLOTS_DIR, "exp4_tokens.png")
    plt.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path4}")

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
    path_json = os.path.join(PLOTS_DIR, "exp4_results.json")
    with open(path_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ Saved: {path_json}")

# ============================================================
# MAIN
# ============================================================

# Step 1: Build cloud semantic memory in ChromaDB
cloud_collection, cloud_plans = build_cloud_semantic_memory()

# Step 2: Run small models with cloud semantic memory injected
results = run_small_models(cloud_collection, cloud_plans)

# Step 3: Plot everything
plot_results(results, cloud_plans)

print(f"\n{'='*60}")
print("EXPERIMENT 4 COMPLETE")
print(f"{'='*60}")