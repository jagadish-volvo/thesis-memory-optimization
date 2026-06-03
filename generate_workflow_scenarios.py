"""
Synthetic Workflow Dataset Generator
=====================================
Generates synthetic workflow execution scenarios for
Experiment 6 — RL-Based Adaptive Coordination.

PA2534 — Master Thesis in Software Engineering
Student: Mani Prabhu Jagadish Vaddiparthi (mava22@student.bth.se)

PURPOSE:
  Creates a fixed synthetic dataset of workflow task sequences
  that both the static baseline and RL agent will be evaluated on.
  Using identical scenarios ensures fair comparison and full
  reproducibility — both agents face the exact same conditions.

  This dataset will be published on Zenodo upon thesis submission
  as specified in thesis proposal §4.3.

DATASET STRUCTURE:
  10 workflow scenarios — each with 200 tasks
  Each scenario represents a different operational condition:

  Scenario 0 — Uniform distribution
    All task types arrive equally (20% each)
    Represents balanced workload — baseline condition

  Scenario 1 — Execution heavy (bottleneck)
    60% execution tasks — represents peak verification load
    This is the primary bottleneck scenario from thesis §1.1

  Scenario 2 — High priority surge
    40% high priority tasks — represents urgent verification deadline
    Tests how coordination handles priority pressure

  Scenario 3 — Planning heavy
    50% planning tasks — early project phase simulation
    Tests routing when a different agent becomes bottleneck

  Scenario 4 — Monitoring heavy
    50% monitoring tasks — active testing phase simulation
    Tests routing under monitoring agent overload

  Scenario 5 — Mixed moderate load
    Realistic distribution with moderate arrival rate
    Represents typical day-to-day workflow operation

  Scenario 6 — Burst pattern
    Tasks arrive in bursts — high load then low load
    Tests how RL handles sudden workload spikes

  Scenario 7 — Escalating load
    Task arrival rate increases over time
    Tests RL adaptation as system load grows

  Scenario 8 — Random variation 1
    Random task distribution with fixed seed
    Reproducible random workload pattern

  Scenario 9 — Random variation 2
    Different random distribution with fixed seed
    Second reproducible random workload pattern

Each task has:
  - task_type     : 0=planning, 1=scheduling, 2=execution,
                    3=monitoring, 4=reporting
  - priority      : 0=low, 1=medium, 2=high
  - arrival_time  : when task arrives (seconds from episode start)
  - scenario_id   : which scenario this task belongs to
  - task_id       : unique identifier within scenario

OUTPUT:
  workflow_scenarios.json — complete dataset
  workflow_scenarios_summary.json — statistics summary
"""

import json
import os
import random
import numpy as np
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE    = os.path.join(SCRIPT_DIR, "workflow_scenarios.json")
SUMMARY_FILE   = os.path.join(SCRIPT_DIR, "workflow_scenarios_summary.json")

RANDOM_SEED    = 42
TASKS_PER_SCENARIO = 200
INTER_ARRIVAL  = 1.5   # seconds between task arrivals

# Task type names
TASK_NAMES = {
    0: "planning",
    1: "scheduling",
    2: "execution",
    3: "monitoring",
    4: "reporting"
}

PRIORITY_NAMES = {
    0: "low",
    1: "medium",
    2: "high"
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# SCENARIO DEFINITIONS
# Each scenario defines task type and priority distributions
# ============================================================

SCENARIOS = [
    {
        "id":          0,
        "name":        "Uniform Distribution",
        "description": "All task types arrive equally — balanced workload baseline",
        "task_weights":     [20, 20, 20, 20, 20],  # equal distribution
        "priority_weights": [33, 34, 33],
        "arrival_pattern":  "uniform"
    },
    {
        "id":          1,
        "name":        "Execution Heavy (Bottleneck)",
        "description": "60% execution tasks — primary bottleneck scenario from thesis §1.1",
        "task_weights":     [5, 10, 60, 15, 10],   # execution dominates
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "uniform"
    },
    {
        "id":          2,
        "name":        "High Priority Surge",
        "description": "40% high priority tasks — urgent verification deadline",
        "task_weights":     [5, 10, 60, 15, 10],
        "priority_weights": [20, 40, 40],           # more high priority
        "arrival_pattern":  "uniform"
    },
    {
        "id":          3,
        "name":        "Planning Heavy",
        "description": "50% planning tasks — early project phase simulation",
        "task_weights":     [50, 15, 15, 10, 10],   # planning dominates
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "uniform"
    },
    {
        "id":          4,
        "name":        "Monitoring Heavy",
        "description": "50% monitoring tasks — active testing phase simulation",
        "task_weights":     [5, 10, 15, 50, 20],    # monitoring dominates
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "uniform"
    },
    {
        "id":          5,
        "name":        "Mixed Moderate Load",
        "description": "Realistic distribution — typical day-to-day workflow",
        "task_weights":     [10, 15, 40, 25, 10],
        "priority_weights": [40, 45, 15],
        "arrival_pattern":  "uniform"
    },
    {
        "id":          6,
        "name":        "Burst Pattern",
        "description": "Tasks arrive in bursts — sudden workload spikes",
        "task_weights":     [5, 10, 60, 15, 10],
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "burst"
    },
    {
        "id":          7,
        "name":        "Escalating Load",
        "description": "Task arrival rate increases over time — growing workload",
        "task_weights":     [5, 10, 60, 15, 10],
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "escalating"
    },
    {
        "id":          8,
        "name":        "Random Variation 1",
        "description": "Random task distribution — seed 101",
        "task_weights":     None,   # will be randomised with fixed seed
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "uniform",
        "seed":             101
    },
    {
        "id":          9,
        "name":        "Random Variation 2",
        "description": "Different random distribution — seed 202",
        "task_weights":     None,   # will be randomised with fixed seed
        "priority_weights": [30, 50, 20],
        "arrival_pattern":  "uniform",
        "seed":             202
    },
]

# ============================================================
# TASK GENERATOR
# ============================================================

def generate_scenario_tasks(scenario, num_tasks=TASKS_PER_SCENARIO):
    """
    Generate a fixed sequence of tasks for one scenario.

    Parameters:
        scenario  : scenario definition dict
        num_tasks : number of tasks to generate

    Returns:
        list of task dicts with type, priority, arrival_time
    """
    # Handle random variation scenarios
    if scenario.get("seed"):
        rng = random.Random(scenario["seed"])
        np_rng = np.random.RandomState(scenario["seed"])
        # Generate random task weights for this scenario
        raw_weights = [rng.randint(5, 60) for _ in range(5)]
        total = sum(raw_weights)
        task_weights = [w / total * 100 for w in raw_weights]
    else:
        rng = random.Random(RANDOM_SEED + scenario["id"])
        np_rng = np.random.RandomState(RANDOM_SEED + scenario["id"])
        task_weights = scenario["task_weights"]

    priority_weights = scenario["priority_weights"]
    arrival_pattern  = scenario["arrival_pattern"]

    tasks = []
    current_time = 0.0

    for i in range(num_tasks):

        # ── ARRIVAL TIME ──────────────────────────────────────
        if arrival_pattern == "uniform":
            # Fixed inter-arrival time
            inter_arrival = INTER_ARRIVAL

        elif arrival_pattern == "burst":
            # Tasks arrive in bursts of 10, then pause
            burst_size = 10
            if (i % burst_size) < burst_size // 2:
                inter_arrival = 0.5   # fast burst
            else:
                inter_arrival = 3.0   # pause between bursts

        elif arrival_pattern == "escalating":
            # Inter-arrival decreases over time (more tasks later)
            progress = i / num_tasks
            inter_arrival = max(0.5, INTER_ARRIVAL * (1.5 - progress))

        else:
            inter_arrival = INTER_ARRIVAL

        current_time += inter_arrival

        # ── TASK TYPE ─────────────────────────────────────────
        task_type = rng.choices(
            population=list(range(5)),
            weights=task_weights
        )[0]

        # ── PRIORITY ──────────────────────────────────────────
        priority = rng.choices(
            population=[0, 1, 2],
            weights=priority_weights
        )[0]

        tasks.append({
            "task_id":      i,
            "scenario_id":  scenario["id"],
            "task_type":    task_type,
            "task_name":    TASK_NAMES[task_type],
            "priority":     priority,
            "priority_name": PRIORITY_NAMES[priority],
            "arrival_time": round(current_time, 3)
        })

    return tasks


# ============================================================
# GENERATE ALL SCENARIOS
# ============================================================

print("=" * 60)
print("SYNTHETIC WORKFLOW DATASET GENERATOR")
print("PA2534 — Master Thesis — Mani Prabhu")
print("=" * 60)
print(f"\nGenerating {len(SCENARIOS)} workflow scenarios")
print(f"Tasks per scenario: {TASKS_PER_SCENARIO}")
print(f"Random seed: {RANDOM_SEED}")
print()

all_scenarios = []
summary       = []

for scenario in SCENARIOS:
    tasks = generate_scenario_tasks(scenario, TASKS_PER_SCENARIO)

    # Calculate statistics
    type_counts     = Counter(t["task_type"] for t in tasks)
    priority_counts = Counter(t["priority"]  for t in tasks)

    type_dist = {
        TASK_NAMES[t]: round(type_counts[t] / len(tasks) * 100, 1)
        for t in range(5)
    }
    priority_dist = {
        PRIORITY_NAMES[p]: round(priority_counts[p] / len(tasks) * 100, 1)
        for p in range(3)
    }

    scenario_data = {
        "scenario_id":   scenario["id"],
        "scenario_name": scenario["name"],
        "description":   scenario["description"],
        "num_tasks":     len(tasks),
        "task_distribution": type_dist,
        "priority_distribution": priority_dist,
        "tasks": tasks
    }

    all_scenarios.append(scenario_data)

    # Print summary
    print(f"  Scenario {scenario['id']:2d}: {scenario['name']}")
    print(f"    {scenario['description']}")
    print(f"    Tasks: {len(tasks)}")
    print(f"    Type distribution:")
    for name, pct in type_dist.items():
        bar = "█" * int(pct / 5)
        print(f"      {name:12s}: {pct:5.1f}% {bar}")
    print(f"    Priority: low={priority_dist['low']}%  "
          f"medium={priority_dist['medium']}%  "
          f"high={priority_dist['high']}%")
    print()

    summary.append({
        "scenario_id":           scenario["id"],
        "scenario_name":         scenario["name"],
        "description":           scenario["description"],
        "num_tasks":             len(tasks),
        "task_distribution":     type_dist,
        "priority_distribution": priority_dist
    })

# ============================================================
# SAVE DATASET
# ============================================================

dataset = {
    "metadata": {
        "title":       "Synthetic Workflow Execution Dataset",
        "thesis":      "PA2534 — Performance Optimization of Agentic Automation",
        "student":     "Mani Prabhu Jagadish Vaddiparthi",
        "institution": "Blekinge Institute of Technology (BTH)",
        "company":     "Volvo Group",
        "purpose":     "RL-based adaptive coordination evaluation",
        "random_seed": RANDOM_SEED,
        "num_scenarios": len(SCENARIOS),
        "tasks_per_scenario": TASKS_PER_SCENARIO,
        "total_tasks": len(SCENARIOS) * TASKS_PER_SCENARIO,
        "inter_arrival_time_seconds": INTER_ARRIVAL,
        "task_types": TASK_NAMES,
        "priority_levels": PRIORITY_NAMES,
        "agent_specialisation": {
            "Agent 0 Planning":   "specialist in planning tasks",
            "Agent 1 Scheduling": "specialist in scheduling tasks",
            "Agent 2 Execution":  "specialist in execution tasks",
            "Agent 3 Monitoring": "specialist in monitoring tasks",
            "Agent 4 Reporting":  "specialist in reporting tasks"
        },
        "processing_times": {
            "specialist_agent_seconds":     2,
            "non_specialist_agent_seconds": 5,
            "justification": (
                "Specialist agents process primary task type at 2.5x "
                "speed reflecting productivity advantage of specialisation "
                "[Wooldridge 2009, Multi-Agent Systems]"
            )
        }
    },
    "scenarios": all_scenarios
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(dataset, f, indent=2)

with open(SUMMARY_FILE, "w") as f:
    json.dump({
        "metadata": dataset["metadata"],
        "scenarios": summary
    }, f, indent=2)

# Calculate file size
size_kb = os.path.getsize(OUTPUT_FILE) / 1024

print("=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)
print(f"\n  Scenarios generated : {len(SCENARIOS)}")
print(f"  Tasks per scenario  : {TASKS_PER_SCENARIO}")
print(f"  Total tasks         : {len(SCENARIOS) * TASKS_PER_SCENARIO}")
print(f"\n  Output files:")
print(f"    {OUTPUT_FILE}")
print(f"    ({size_kb:.1f} KB)")
print(f"    {SUMMARY_FILE}")
print(f"\n  This dataset will be used by experiment6_rl.py")
print(f"  Both static and RL agents evaluated on identical scenarios")
print(f"  Ensures fair comparison and full reproducibility")
print(f"\n  Ready to publish on Zenodo upon thesis submission")
print("=" * 60)
