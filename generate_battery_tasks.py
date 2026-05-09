"""
Battery Task Generator for Memory Module Experiments (Experiments 3 and 4)
==========================================================================
Generates 20 diverse battery verification and engineering tasks
with predefined keyword groups for step coverage evaluation.

All tasks are within the battery/energy storage domain to ensure
semantic memory retrieval works correctly without cross-domain contamination.

Each task has 5 keyword groups with synonyms.
Step coverage = how many of the 5 groups appear in the generated answer.

Task categories (4 each):
  - Thermal management    : overheating, cooling, thermal runaway
  - BMS validation        : battery management system testing
  - Cell and module       : cell-level diagnosis and evaluation
  - Safety and protection : safety validation and fault protection
  - Performance and life  : cycle life, capacity, degradation

Output: battery_tasks.json
"""

import json
import os

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "battery_tasks.json")

# ============================================================
# 20 BATTERY VERIFICATION TASKS WITH KEYWORD GROUPS
# 4 tasks per category x 5 categories = 20 tasks total
# Each task has exactly 5 keyword groups with 4-5 synonyms
# ============================================================

TASKS = [

    # ── THERMAL MANAGEMENT (4) ───────────────────────────────
    {
        "task": "Diagnose why a lithium-ion battery pack overheats during fast charging and propose corrective actions.",
        "category": "thermal",
        "expected_keywords": [
            ["temperature", "thermal", "heat", "overheating", "temp"],
            ["cooling", "coolant", "heat dissipation", "thermal management", "heat sink"],
            ["charging", "charge rate", "c-rate", "fast charge", "current"],
            ["resistance", "impedance", "internal resistance", "joule heating", "ir drop"],
            ["bms", "battery management", "protection circuit", "thermal cutoff", "limit"]
        ]
    },
    {
        "task": "Design a thermal management system for a high-voltage battery pack used in electric commercial vehicles.",
        "category": "thermal",
        "expected_keywords": [
            ["cooling", "liquid cooling", "air cooling", "thermal management", "heat exchanger"],
            ["temperature", "temperature uniformity", "gradient", "distribution", "setpoint"],
            ["sensor", "thermocouple", "thermistor", "temperature sensor", "monitoring"],
            ["pump", "coolant pump", "flow rate", "circulation", "coolant"],
            ["control", "controller", "pid", "regulation", "feedback"]
        ]
    },
    {
        "task": "Evaluate the risk of thermal runaway propagation in a prismatic cell battery module and propose mitigation strategies.",
        "category": "thermal",
        "expected_keywords": [
            ["thermal runaway", "propagation", "cascade", "runaway", "exothermic"],
            ["temperature", "heat generation", "self-heating", "peak temperature", "rise"],
            ["venting", "gas", "electrolyte", "decomposition", "smoke"],
            ["separator", "electrode", "cell failure", "short circuit", "internal short"],
            ["mitigation", "fireproof", "barrier", "suppression", "protection"]
        ]
    },
    {
        "task": "Plan a thermal cycling test procedure to evaluate the durability of a battery pack's thermal management system under extreme temperature conditions.",
        "category": "thermal",
        "expected_keywords": [
            ["thermal cycling", "temperature cycling", "cycle", "hot", "cold"],
            ["chamber", "environmental chamber", "climate", "conditioning", "soak"],
            ["expansion", "contraction", "stress", "fatigue", "mechanical"],
            ["seal", "gasket", "connector", "leakage", "integrity"],
            ["performance", "capacity", "degradation", "retention", "measurement"]
        ]
    },

    # ── BMS VALIDATION (4) ───────────────────────────────────
    {
        "task": "Plan a safety validation procedure for a battery management system to verify overvoltage and undervoltage protection.",
        "category": "bms",
        "expected_keywords": [
            ["overvoltage", "overvoltage protection", "voltage limit", "overcharge", "upper cutoff"],
            ["undervoltage", "undervoltage protection", "deep discharge", "lower cutoff", "undercharge"],
            ["protection", "cutoff", "disconnect", "relay", "contactor"],
            ["test", "validation", "verification", "simulate", "inject"],
            ["bms", "battery management system", "controller", "logic", "firmware"]
        ]
    },
    {
        "task": "Design a test procedure to validate the state of charge estimation accuracy of a battery management system across different temperatures.",
        "category": "bms",
        "expected_keywords": [
            ["soc", "state of charge", "estimation", "accuracy", "error"],
            ["algorithm", "kalman filter", "coulomb counting", "ocv", "model"],
            ["temperature", "low temperature", "high temperature", "ambient", "thermal"],
            ["reference", "coulombic", "charge", "discharge", "benchmark"],
            ["validation", "test", "deviation", "tolerance", "measurement"]
        ]
    },
    {
        "task": "Evaluate the communication reliability of a BMS CAN bus interface under electromagnetic interference conditions in an industrial environment.",
        "category": "bms",
        "expected_keywords": [
            ["can bus", "communication", "protocol", "interface", "bus"],
            ["emi", "electromagnetic", "interference", "noise", "disturbance"],
            ["message", "frame", "error", "fault", "transmission"],
            ["shield", "grounding", "filtering", "termination", "isolation"],
            ["reliability", "test", "validation", "robustness", "immunity"]
        ]
    },
    {
        "task": "Develop a validation procedure for the cell balancing algorithm in a battery management system to ensure uniform state of charge across all cells.",
        "category": "bms",
        "expected_keywords": [
            ["balancing", "cell balancing", "active balancing", "passive balancing", "equalization"],
            ["soc", "state of charge", "voltage", "uniformity", "imbalance"],
            ["algorithm", "balancing algorithm", "logic", "trigger", "threshold"],
            ["dissipation", "resistor", "energy transfer", "shunting", "bypass"],
            ["test", "validation", "verification", "measurement", "monitoring"]
        ]
    },

    # ── CELL AND MODULE (4) ──────────────────────────────────
    {
        "task": "Identify the root cause of capacity fade in a lithium-ion battery cell after 500 charge-discharge cycles and recommend corrective actions.",
        "category": "cell",
        "expected_keywords": [
            ["capacity", "capacity fade", "degradation", "retention", "loss"],
            ["cycle", "cycling", "charge discharge", "depth of discharge", "dod"],
            ["sei", "solid electrolyte interphase", "film", "growth", "lithium plating"],
            ["electrolyte", "decomposition", "oxidation", "solvent", "salt"],
            ["impedance", "internal resistance", "resistance growth", "eis", "spectroscopy"]
        ]
    },
    {
        "task": "Design a formation cycling procedure for new lithium-ion cells to optimize initial SEI layer formation and maximize long-term cycle life.",
        "category": "cell",
        "expected_keywords": [
            ["formation", "formation cycling", "sei", "initial cycle", "first cycle"],
            ["current", "charge rate", "c-rate", "low current", "formation current"],
            ["voltage", "cutoff voltage", "upper limit", "lower limit", "formation voltage"],
            ["capacity", "coulombic efficiency", "irreversible", "first cycle loss", "charge"],
            ["temperature", "formation temperature", "ambient", "controlled", "conditioning"]
        ]
    },
    {
        "task": "Evaluate the impact of high ambient temperature storage on the self-discharge rate and capacity retention of lithium iron phosphate cells.",
        "category": "cell",
        "expected_keywords": [
            ["storage", "shelf life", "self discharge", "calendar aging", "aging"],
            ["temperature", "high temperature", "ambient", "storage temperature", "thermal"],
            ["capacity", "retention", "loss", "degradation", "fade"],
            ["voltage", "open circuit voltage", "ocv", "recovery", "resting"],
            ["lfp", "lithium iron phosphate", "cathode", "chemistry", "cell"]
        ]
    },
    {
        "task": "Diagnose the root cause of voltage imbalance across cells in a series-connected battery module and propose a corrective maintenance plan.",
        "category": "cell",
        "expected_keywords": [
            ["voltage", "imbalance", "cell voltage", "deviation", "variation"],
            ["capacity", "capacity mismatch", "aging", "degradation", "difference"],
            ["self discharge", "leakage", "parasitic", "current", "rate"],
            ["balancing", "active balancing", "passive balancing", "equalization", "bms"],
            ["inspection", "measurement", "eis", "internal resistance", "diagnosis"]
        ]
    },

    # ── SAFETY AND PROTECTION (4) ────────────────────────────
    {
        "task": "Plan a nail penetration and crush test procedure to evaluate the abuse tolerance and safety response of a pouch lithium-ion cell.",
        "category": "safety",
        "expected_keywords": [
            ["nail penetration", "crush", "abuse", "mechanical", "short circuit"],
            ["thermal runaway", "temperature", "venting", "smoke", "fire"],
            ["safety", "protection", "response", "trigger", "activation"],
            ["monitoring", "sensor", "thermocouple", "voltage", "current"],
            ["standard", "iec", "un", "test procedure", "certification"]
        ]
    },
    {
        "task": "Develop a high-voltage isolation test procedure for an electric vehicle battery pack to verify electrical safety and insulation integrity.",
        "category": "safety",
        "expected_keywords": [
            ["isolation", "insulation", "resistance", "high voltage", "dielectric"],
            ["leakage", "leakage current", "ground fault", "earth", "fault"],
            ["hipot", "hi-pot", "withstand", "test voltage", "breakdown"],
            ["measurement", "megohmmeter", "resistance measurement", "impedance", "monitoring"],
            ["safety", "standard", "iso", "iec", "compliance"]
        ]
    },
    {
        "task": "Design an overtemperature protection strategy for a battery energy storage system to prevent thermal runaway in industrial applications.",
        "category": "safety",
        "expected_keywords": [
            ["overtemperature", "temperature threshold", "cutoff temperature", "limit", "protection"],
            ["sensor", "thermocouple", "thermistor", "monitoring", "detection"],
            ["shutdown", "disconnect", "contactor", "relay", "isolation"],
            ["cooling", "emergency cooling", "thermal management", "heat removal", "dissipation"],
            ["bms", "protection circuit", "safety system", "redundancy", "failsafe"]
        ]
    },
    {
        "task": "Evaluate the short circuit protection capability of a battery management system under both internal and external short circuit conditions.",
        "category": "safety",
        "expected_keywords": [
            ["short circuit", "external short", "internal short", "short", "fault"],
            ["current", "peak current", "fault current", "overcurrent", "surge"],
            ["protection", "fuse", "contactor", "mosfet", "switch"],
            ["response time", "detection time", "reaction", "cutoff", "speed"],
            ["temperature", "heat", "thermal", "dissipation", "rating"]
        ]
    },

    # ── PERFORMANCE AND LIFE (4) ─────────────────────────────
    {
        "task": "Design a cycle life verification test plan for a lithium-ion battery cell intended for grid energy storage applications.",
        "category": "performance",
        "expected_keywords": [
            ["cycle life", "cycling", "cycles", "end of life", "eol"],
            ["capacity", "retention", "fade", "degradation", "threshold"],
            ["depth of discharge", "dod", "charge", "discharge", "protocol"],
            ["temperature", "ambient temperature", "controlled", "chamber", "condition"],
            ["measurement", "reference performance test", "rpt", "characterization", "data"]
        ]
    },
    {
        "task": "Evaluate the rate capability of a lithium-ion battery cell at different discharge rates and temperatures for electric vehicle applications.",
        "category": "performance",
        "expected_keywords": [
            ["rate capability", "c-rate", "discharge rate", "power", "current"],
            ["capacity", "delivered capacity", "energy", "efficiency", "output"],
            ["temperature", "low temperature", "performance", "cold", "ambient"],
            ["voltage", "voltage sag", "polarization", "internal resistance", "drop"],
            ["characterization", "test", "measurement", "galvanostatic", "protocol"]
        ]
    },
    {
        "task": "Plan a calendar aging study to predict the long-term capacity retention of lithium-ion cells stored at various state of charge and temperature conditions.",
        "category": "performance",
        "expected_keywords": [
            ["calendar aging", "storage aging", "aging", "shelf", "time"],
            ["soc", "state of charge", "storage soc", "level", "condition"],
            ["temperature", "storage temperature", "elevated", "ambient", "condition"],
            ["capacity", "retention", "fade", "loss", "measurement"],
            ["prediction", "model", "arrhenius", "lifetime", "projection"]
        ]
    },
    {
        "task": "Diagnose why a battery pack used in a forklift application shows accelerated capacity degradation within six months of deployment and propose solutions.",
        "category": "performance",
        "expected_keywords": [
            ["degradation", "capacity loss", "accelerated", "aging", "fade"],
            ["charging", "opportunity charging", "partial charge", "overcharge", "protocol"],
            ["temperature", "operating temperature", "heat", "thermal stress", "ambient"],
            ["depth of discharge", "dod", "cycling", "stress", "utilization"],
            ["maintenance", "equalization", "balancing", "replacement", "recommendation"]
        ]
    },
]

# ============================================================
# VALIDATE AND SAVE
# ============================================================

print("=" * 60)
print("BATTERY TASK GENERATOR — Memory Module Experiments 3 & 4")
print("=" * 60)
print(f"Total tasks: {len(TASKS)}")
print(f"Categories:")
from collections import Counter
cats = Counter(t["category"] for t in TASKS)
for cat, count in cats.items():
    print(f"  {cat}: {count} tasks")

# Validate each task has exactly 5 keyword groups
issues = []
for i, t in enumerate(TASKS):
    if len(t["expected_keywords"]) != 5:
        issues.append(f"Task {i+1} has {len(t['expected_keywords'])} keyword groups (expected 5)")
    for j, group in enumerate(t["expected_keywords"]):
        if len(group) < 3:
            issues.append(f"Task {i+1} group {j+1} has only {len(group)} keywords")

if issues:
    print(f"\n⚠️  VALIDATION ISSUES:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"\n✅ All {len(TASKS)} tasks validated — each has exactly 5 keyword groups")

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(TASKS, f, indent=2)

print(f"✅ Saved to: {OUTPUT_FILE}")
print(f"\nSample task:")
print(f"  [{TASKS[0]['category']}] {TASKS[0]['task'][:75]}...")
print(f"  Keywords group 1: {TASKS[0]['expected_keywords'][0]}")
print("=" * 60)
print("Ready to use in Experiments 3 and 4.")
print("=" * 60)
