"""
Task Generator for AdaptEvolve Experiment
==========================================
Generates 100 diverse industrial tasks across 4 categories:
  - 25 Simple questions
  - 25 Medium complexity questions
  - 25 Complex long tasks
  - 25 Coding explanation tasks

Uses Ollama (qwen3.5:4b) to generate tasks automatically.
Saves to tasks.json for use in the AdaptEvolve experiment.
"""

import requests
import json
import os
import time

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL  = "http://localhost:11434/api/chat"
GEN_MODEL   = "qwen3.5:4b"   # use 4b for good quality task generation

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "tasks.json")

# ============================================================
# TASK CATEGORIES — 25 tasks each = 100 total
# ============================================================

CATEGORIES = [
    {
        "name": "simple",
        "label": "Simple Questions",
        "count": 25,
        "description": (
            "Simple, short factual questions about industrial engineering, "
            "manufacturing, automation, or industrial processes. "
            "Each question should be answerable in 1-3 sentences. "
            "Examples: definitions, basic concepts, naming components."
        ),
        "example": "What does PLC stand for in industrial automation?"
    },
    {
        "name": "medium",
        "label": "Medium Complexity",
        "count": 25,
        "description": (
            "Medium complexity questions about industrial systems, processes, "
            "or engineering concepts that require explanation and some reasoning. "
            "Each question should require a paragraph or a few steps to answer. "
            "Examples: how-things-work, comparisons, cause-and-effect explanations."
        ),
        "example": "Explain how a PID controller maintains stable temperature in an industrial furnace."
    },
    {
        "name": "complex",
        "label": "Complex Long Tasks",
        "count": 25,
        "description": (
            "Complex, multi-step industrial engineering tasks that require "
            "detailed planning, diagnosis, or procedure design. "
            "Each task should require at least 5-6 steps to answer properly. "
            "Examples: design procedures, diagnose faults, plan validations, evaluate systems."
        ),
        "example": "Design a preventive maintenance plan for a high-speed CNC milling machine used in automotive part manufacturing."
    },
    {
        "name": "coding",
        "label": "Coding Explanation Tasks",
        "count": 25,
        "description": (
            "Tasks that ask the model to describe or explain the logic, steps, "
            "and approach to implement a software solution for an industrial problem. "
            "Do NOT ask the model to write actual runnable code — ask it to describe "
            "the implementation approach, key steps, data structures, and logic in text. "
            "Examples: describe how to implement algorithms, explain data processing pipelines, "
            "outline the logic for industrial software systems."
        ),
        "example": "Describe the steps and logic to implement a Python-based anomaly detection system for vibration sensor data collected from rotating industrial machinery."
    },
]

# ============================================================
# GENERATION
# ============================================================

def generate_text(prompt, max_tokens=2000):
    """Generate text using Ollama."""
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.8, "num_predict": max_tokens}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  [ERROR] {e}")
        return ""

def generate_batch(category, batch_num, batch_size=5):
    """
    Generate a small batch of tasks for a category.
    Retries up to 3 times if it fails or returns too few tasks.
    """
    import re
    name    = category["label"]
    desc    = category["description"]
    example = category["example"]

    prompt = f"""Generate exactly {batch_size} diverse industrial tasks or questions.

Category: {name}
Description: {desc}
Example: {example}

Requirements:
- Each task must be on its own numbered line (1. 2. 3. etc.)
- Tasks must cover DIFFERENT industrial domains — include variety from:
  manufacturing, automation, robotics, quality control, safety systems,
  process engineering, supply chain, maintenance, industrial IoT,
  SCADA systems, hydraulics, pneumatics, welding, CNC machining,
  electrical systems, and industrial software
- Do NOT repeat similar tasks
- Do NOT include battery or energy storage tasks
- Output ONLY the numbered list, no headers or extra text

Generate {batch_size} tasks now:"""

    for attempt in range(3):  # retry up to 3 times
        response = generate_text(prompt, max_tokens=600)
        if not response:
            print(f" [retry {attempt+1}]", end="")
            time.sleep(5)
            continue

        tasks = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r'^\d+[\.\)\:]\s*', '', line).strip()
            if len(cleaned) > 20:
                tasks.append(cleaned)

        tasks = tasks[:batch_size]
        if len(tasks) >= batch_size:
            return tasks
        else:
            print(f" [only {len(tasks)}, retry {attempt+1}]", end="")
            time.sleep(5)

    return tasks  # return whatever we got after 3 attempts

def generate_tasks_for_category(category):
    """
    Generate tasks for a single category in small batches of 5.
    5 batches x 5 tasks = 25 tasks total per category.
    No timeout risk.
    """
    count      = category["count"]
    name       = category["label"]
    batch_size = 5
    num_batches = count // batch_size

    print(f"\n  Generating {count} {name} tasks ({num_batches} batches of {batch_size})...")

    all_tasks = []
    for i in range(num_batches):
        print(f"    Batch {i+1}/{num_batches}...", end=" ")
        batch = generate_batch(category, i, batch_size)
        all_tasks.extend(batch)
        print(f"got {len(batch)} tasks (total: {len(all_tasks)})")
        time.sleep(4)  # pause between batches to avoid overloading

    print(f"  ✅ Generated {len(all_tasks)} tasks")
    return all_tasks[:count]

# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("TASK GENERATOR — AdaptEvolve Experiment")
print("=" * 60)
print(f"Model: {GEN_MODEL}")
print(f"Output: {OUTPUT_FILE}")
print(f"Target: 100 tasks across 4 categories")

all_tasks = []

for category in CATEGORIES:
    tasks = generate_tasks_for_category(category)

    for task_text in tasks:
        all_tasks.append({
            "task": task_text,
            "category": category["name"],
            "category_label": category["label"]
        })

print(f"\n{'=' * 60}")
print(f"Total tasks generated: {len(all_tasks)}")
print(f"  Simple:   {sum(1 for t in all_tasks if t['category'] == 'simple')}")
print(f"  Medium:   {sum(1 for t in all_tasks if t['category'] == 'medium')}")
print(f"  Complex:  {sum(1 for t in all_tasks if t['category'] == 'complex')}")
print(f"  Coding:   {sum(1 for t in all_tasks if t['category'] == 'coding')}")

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_tasks, f, indent=2)

print(f"\n✅ Saved to: {OUTPUT_FILE}")
print("=" * 60)
print("Ready to use in AdaptEvolve experiment.")
print("=" * 60)