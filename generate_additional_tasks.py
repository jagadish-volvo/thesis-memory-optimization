"""
Additional Task Generator for AdaptEvolve Experiment 5
=======================================================
Generates 100 additional diverse general industrial engineering tasks
to extend tasks.json from 100 to 200 tasks total.

Same 4 categories as original:
  - simple  : 25 tasks
  - medium  : 25 tasks
  - complex : 25 tasks
  - coding  : 25 tasks

All tasks are NEW and non-overlapping with the original 100.
Output: appends to existing tasks.json → 200 tasks total.
"""

import requests
import json
import os
import time

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL  = "http://localhost:11434/api/chat"
GEN_MODEL   = "qwen3.5:4b"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TASKS_FILE  = os.path.join(SCRIPT_DIR, "tasks.json")

# ============================================================
# 100 NEW TASKS — HARDCODED FOR RELIABILITY
# Same format as original tasks.json
# 25 simple, 25 medium, 25 complex, 25 coding
# All topics are different from the original 100
# ============================================================

NEW_TASKS = [

    # ── SIMPLE (25) ─────────────────────────────────────────

    {"task": "What is the function of a pressure relief valve in a hydraulic system?", "category": "simple"},
    {"task": "Define the term 'cycle time' in the context of an automated assembly line.", "category": "simple"},
    {"task": "What does OEE stand for and what does it measure in manufacturing?", "category": "simple"},
    {"task": "Explain the purpose of a watchdog timer in an embedded industrial control system.", "category": "simple"},
    {"task": "What is the difference between a servo motor and a stepper motor in industrial automation?", "category": "simple"},
    {"task": "Define 'dead band' in the context of a PID controller.", "category": "simple"},
    {"task": "What is the role of a fieldbus protocol such as PROFIBUS in factory automation?", "category": "simple"},
    {"task": "Explain what a P&ID diagram is and what information it contains.", "category": "simple"},
    {"task": "What is the difference between preventive maintenance and corrective maintenance?", "category": "simple"},
    {"task": "Define 'torque ripple' and explain why it matters in precision motor control.", "category": "simple"},
    {"task": "What is a safety integrity level (SIL) and how is it used in industrial safety?", "category": "simple"},
    {"task": "Explain what ANDON is in a lean manufacturing context.", "category": "simple"},
    {"task": "What is the purpose of a flow meter in an industrial process control system?", "category": "simple"},
    {"task": "Define 'jidoka' and explain its significance in Toyota Production System.", "category": "simple"},
    {"task": "What is the difference between an encoder and a resolver in motion control?", "category": "simple"},
    {"task": "Explain what a Poka-Yoke device is and give an industrial example.", "category": "simple"},
    {"task": "What is the role of a distributed control system (DCS) in process industries?", "category": "simple"},
    {"task": "Define 'mean time between failures' (MTBF) and explain how it is calculated.", "category": "simple"},
    {"task": "What is the purpose of a check valve in a pneumatic circuit?", "category": "simple"},
    {"task": "Explain what a Kanban system is and how it controls inventory in manufacturing.", "category": "simple"},
    {"task": "What is the difference between open-loop and closed-loop control in industrial systems?", "category": "simple"},
    {"task": "Define 'backlash' in mechanical systems and explain its effect on positioning accuracy.", "category": "simple"},
    {"task": "What is a safe torque off (STO) function in variable frequency drives?", "category": "simple"},
    {"task": "Explain what a programmable safety controller (PSC) does in a safety system.", "category": "simple"},
    {"task": "What is the purpose of an industrial Ethernet switch in a factory network?", "category": "simple"},

    # ── MEDIUM (25) ─────────────────────────────────────────

    {"task": "Explain how a proportional-integral-derivative (PID) controller works and describe the effect of each tuning parameter on system response.", "category": "medium"},
    {"task": "Describe the working principle of a variable frequency drive (VFD) and explain how it reduces energy consumption in pump applications.", "category": "medium"},
    {"task": "Compare the advantages and disadvantages of pneumatic versus electric actuation systems in high-speed assembly applications.", "category": "medium"},
    {"task": "Explain how condition monitoring using vibration analysis can detect early bearing failure in rotating machinery.", "category": "medium"},
    {"task": "Describe the key differences between OPC-UA and MQTT protocols and explain which is more suitable for industrial IoT applications.", "category": "medium"},
    {"task": "Explain how a digital twin differs from a simulation model and describe the key data requirements for building an accurate digital twin.", "category": "medium"},
    {"task": "Describe the steps involved in commissioning a new industrial robot in a manufacturing cell, including safety validation.", "category": "medium"},
    {"task": "Explain what edge computing means in an industrial context and describe its advantages over cloud-only architectures for real-time control.", "category": "medium"},
    {"task": "Describe how statistical process control (SPC) charts are used to monitor and maintain product quality in a continuous manufacturing process.", "category": "medium"},
    {"task": "Explain the difference between functional safety and cybersecurity in industrial control systems and describe why both are required.", "category": "medium"},
    {"task": "Describe how a SCADA system collects, processes, and displays data from field devices in a water treatment facility.", "category": "medium"},
    {"task": "Explain what machine vision is in industrial quality control and describe how it detects surface defects on high-speed production lines.", "category": "medium"},
    {"task": "Describe the role of IEC 61508 in industrial safety system design and explain its key requirements for hardware and software.", "category": "medium"},
    {"task": "Explain how torque control differs from speed control in servo drive applications and describe when each mode is appropriate.", "category": "medium"},
    {"task": "Describe the working principle of an inductive proximity sensor and explain its advantages over mechanical limit switches in harsh environments.", "category": "medium"},
    {"task": "Explain how a motion profile (trapezoidal or S-curve) improves positioning accuracy and reduces mechanical stress in servo systems.", "category": "medium"},
    {"task": "Describe the key steps in conducting a Failure Mode and Effects Analysis (FMEA) for a new automated welding system.", "category": "medium"},
    {"task": "Explain how a heat exchanger works in an industrial cooling system and describe the factors that determine its thermal efficiency.", "category": "medium"},
    {"task": "Describe what protocol conversion means in industrial automation and explain why it is needed when integrating legacy equipment.", "category": "medium"},
    {"task": "Explain the concept of redundancy in safety-critical industrial systems and describe the difference between hot standby and cold standby configurations.", "category": "medium"},
    {"task": "Describe how a collaborative robot (cobot) differs from a traditional industrial robot and explain the key safety considerations for human-robot collaboration.", "category": "medium"},
    {"task": "Explain how compressed air quality affects pneumatic actuator performance and describe the filtration and drying requirements for different applications.", "category": "medium"},
    {"task": "Describe the purpose of a manufacturing execution system (MES) and explain how it connects ERP systems to shop floor equipment.", "category": "medium"},
    {"task": "Explain what predictive quality means in smart manufacturing and describe how sensor data is used to predict product defects before they occur.", "category": "medium"},
    {"task": "Describe the key differences between ISO 13849 and IEC 62061 safety standards and explain which applies to machinery safety systems.", "category": "medium"},

    # ── COMPLEX (25) ────────────────────────────────────────

    {"task": "Design a comprehensive energy management strategy for a large automotive stamping plant that integrates demand response, peak shaving, and waste heat recovery to reduce total energy costs by 20% over two years.", "category": "complex"},
    {"task": "Develop a multi-layer cybersecurity architecture for an oil refinery SCADA network that segments operational technology from IT systems while maintaining real-time data visibility for remote monitoring.", "category": "complex"},
    {"task": "Create a detailed validation and qualification plan for a new pharmaceutical filling line, covering equipment qualification (IQ/OQ/PQ), process validation, and regulatory compliance with FDA 21 CFR Part 11.", "category": "complex"},
    {"task": "Design a predictive maintenance framework for a fleet of 50 CNC machining centers that integrates vibration, thermal, and acoustic sensors with a machine learning model to predict tool failures 24 hours in advance.", "category": "complex"},
    {"task": "Develop a comprehensive functional safety plan for a new automated guided vehicle (AGV) fleet operating in a mixed human-robot warehouse, covering hazard analysis, SIL assessment, and safety architecture design.", "category": "complex"},
    {"task": "Design a digital twin architecture for a continuous casting process in a steel mill that models solidification dynamics, thermal stress, and surface quality in real-time to optimize casting speed and reduce defect rates.", "category": "complex"},
    {"task": "Create a detailed root cause analysis and corrective action plan for a recurring hydraulic seal failure in a heavy press line, covering tribological analysis, operating condition review, and material selection recommendations.", "category": "complex"},
    {"task": "Develop a comprehensive OEE improvement roadmap for a beverage bottling plant that identifies the top five loss categories, proposes technical solutions for each, and defines KPIs to track progress over 12 months.", "category": "complex"},
    {"task": "Design an industrial IoT platform architecture for a multi-site food processing company that aggregates sensor data from 200 machines across five plants into a unified real-time dashboard with anomaly detection.", "category": "complex"},
    {"task": "Create a detailed risk assessment and mitigation plan for introducing a collaborative robot to a manual assembly station where workers perform precision insertion tasks with fragile electronic components.", "category": "complex"},
    {"task": "Develop a comprehensive change management procedure for upgrading the firmware of 30 PLCs controlling a continuous chemical production line without interrupting production or compromising process safety.", "category": "complex"},
    {"task": "Design a closed-loop quality control system for a high-speed injection molding process that uses in-mold pressure sensors and cavity temperature data to automatically adjust process parameters and reject defective parts.", "category": "complex"},
    {"task": "Create a detailed commissioning and acceptance testing procedure for a new 10 MW solar-plus-storage microgrid powering an industrial campus, covering grid-forming control, islanding detection, and protection coordination.", "category": "complex"},
    {"task": "Develop a supply chain resilience strategy for a tier-one automotive parts manufacturer that identifies single-source dependencies, proposes dual-sourcing plans, and defines buffer stock levels for critical components.", "category": "complex"},
    {"task": "Design a comprehensive asset lifecycle management plan for a fleet of 100 industrial compressors in a gas processing facility, covering condition monitoring, planned replacement scheduling, and spare parts optimization.", "category": "complex"},
    {"task": "Create a detailed process hazard analysis (PHA) for a new exothermic batch reactor in a specialty chemicals plant, covering runaway reaction scenarios, relief valve sizing, and emergency response procedures.", "category": "complex"},
    {"task": "Develop a multi-site manufacturing network optimization plan for a consumer electronics company that balances production capacity, logistics costs, and lead times across factories in three countries.", "category": "complex"},
    {"task": "Design a comprehensive water treatment and recycling system for a semiconductor fabrication facility that minimizes ultrapure water consumption and meets wastewater discharge regulations.", "category": "complex"},
    {"task": "Create a detailed production scheduling optimization strategy for a mixed-model automotive assembly line that minimizes changeover time, balances workstation utilization, and meets daily customer delivery targets.", "category": "complex"},
    {"task": "Develop a comprehensive alarm management improvement plan for a petrochemical control room that reduces nuisance alarms by 60%, implements alarm prioritization, and trains operators on revised response procedures.", "category": "complex"},
    {"task": "Design a real-time quality monitoring system for a continuous paper manufacturing process that integrates basis weight, moisture, and caliper sensors to automatically adjust forming and pressing parameters.", "category": "complex"},
    {"task": "Create a detailed technology roadmap for implementing Industry 4.0 capabilities in a traditional machined parts factory over five years, covering connectivity infrastructure, data analytics, and workforce upskilling.", "category": "complex"},
    {"task": "Develop a comprehensive emission monitoring and reduction plan for a cement kiln that integrates continuous emissions monitoring systems (CEMS) with combustion optimization algorithms to meet NOx and SOx limits.", "category": "complex"},
    {"task": "Design a fault-tolerant control architecture for a critical pumping station in a water distribution network that maintains flow continuity during single-point failures of sensors, controllers, or actuators.", "category": "complex"},
    {"task": "Create a detailed qualification and revalidation strategy for a sterile manufacturing cleanroom HVAC system, covering differential pressure mapping, particle count verification, and microbial monitoring protocols.", "category": "complex"},

    # ── CODING (25) ─────────────────────────────────────────

    {"task": "Describe the logical approach for implementing a real-time anomaly detection system for industrial sensor streams using a sliding window statistical model to flag deviations beyond three standard deviations.", "category": "coding"},
    {"task": "Explain the data pipeline architecture and processing steps needed to build a digital twin that mirrors the state of a physical conveyor system using OPC-UA data feeds.", "category": "coding"},
    {"task": "Outline the algorithm design and key data structures required to implement an automated shift scheduling system for a 24-hour manufacturing plant with variable skill requirements across workstations.", "category": "coding"},
    {"task": "Describe the system architecture and integration approach for connecting a legacy PLC-based production line to a modern cloud analytics platform using an edge gateway and MQTT protocol.", "category": "coding"},
    {"task": "Explain the reasoning and data structures needed to implement a just-in-time material replenishment system that triggers purchase orders based on real-time consumption rates and supplier lead times.", "category": "coding"},
    {"task": "Describe the approach for designing a multi-threaded data acquisition system that simultaneously reads from 100 industrial sensors at different sampling rates without data loss or synchronization errors.", "category": "coding"},
    {"task": "Outline the logic and control flow for implementing an automated tool change sequence in a CNC machining center that verifies tool presence, measures tool length offset, and updates the CNC program accordingly.", "category": "coding"},
    {"task": "Explain the algorithm design and decision logic for a dynamic routing system in a warehouse AMR fleet that minimizes travel distance while avoiding conflicts between multiple robots operating simultaneously.", "category": "coding"},
    {"task": "Describe the architecture and key processing stages of a real-time energy monitoring system that tracks consumption per machine, detects idle waste, and generates automated alerts for energy anomalies.", "category": "coding"},
    {"task": "Outline the data model and processing logic required to implement a traceability system for automotive components that records every manufacturing step, operator, and machine involved from raw material to finished part.", "category": "coding"},
    {"task": "Explain the algorithmic approach for implementing a model predictive control (MPC) system for a batch chemical reactor that optimizes temperature and feed rate profiles to maximize yield while respecting safety constraints.", "category": "coding"},
    {"task": "Describe the system design and communication protocol stack required to implement a wireless sensor network for structural health monitoring of a large industrial building using LoRaWAN.", "category": "coding"},
    {"task": "Outline the reasoning and key components needed to build a computer vision pipeline that detects and classifies weld defects in real-time on a robotic welding production line.", "category": "coding"},
    {"task": "Explain the logical flow and data structures required to implement an automated production reporting system that aggregates shift data from multiple PLCs and generates KPI dashboards for plant management.", "category": "coding"},
    {"task": "Describe the approach for designing a simulation framework to test and validate PLC ladder logic programs for a safety-critical conveyor system before deployment on the production line.", "category": "coding"},
    {"task": "Outline the algorithm and sensor fusion strategy for implementing a force-torque controlled assembly process where a robot must insert a connector with a defined insertion force profile.", "category": "coding"},
    {"task": "Explain the architecture and data flow of a manufacturing analytics platform that correlates machine downtime events with upstream process parameters to identify root causes of recurring failures.", "category": "coding"},
    {"task": "Describe the logical steps and data structures needed to implement an automated calibration management system that tracks calibration due dates, generates work orders, and records results for audit compliance.", "category": "coding"},
    {"task": "Outline the approach for building a reinforcement learning environment that trains an agent to optimize furnace temperature setpoints in a heat treatment process to minimize energy use while meeting metallurgical specifications.", "category": "coding"},
    {"task": "Explain the system architecture and key algorithms needed to implement a real-time tool wear monitoring system for a milling machine using acoustic emission signals and frequency domain analysis.", "category": "coding"},
    {"task": "Describe the reasoning and control logic for implementing an automated leak detection system in a compressed air network that uses flow balance calculations and pressure decay analysis to localize leak sources.", "category": "coding"},
    {"task": "Outline the data architecture and processing pipeline for a quality management system that integrates inspection results from multiple measurement stations to calculate real-time process capability indices.", "category": "coding"},
    {"task": "Explain the approach for designing a federated learning system that trains a predictive maintenance model across multiple factory sites without sharing raw sensor data between sites.", "category": "coding"},
    {"task": "Describe the logical flow and integration points needed to implement an automated non-conformance management system that triggers corrective action workflows when defect rates exceed defined thresholds.", "category": "coding"},
    {"task": "Outline the algorithm design and communication architecture for a real-time production leveling system that dynamically rebalances work orders across parallel manufacturing cells based on current queue lengths and machine availability.", "category": "coding"},
]

# ============================================================
# LOAD EXISTING TASKS AND APPEND
# ============================================================

print("=" * 60)
print("ADDITIONAL TASK GENERATOR — Experiment 5 AdaptEvolve")
print("=" * 60)

# Load existing tasks
if not os.path.exists(TASKS_FILE):
    print(f"ERROR: {TASKS_FILE} not found.")
    print("Please ensure tasks.json exists from the original generate_tasks.py run.")
    exit(1)

with open(TASKS_FILE) as f:
    existing_tasks = json.load(f)

print(f"Existing tasks loaded: {len(existing_tasks)}")

from collections import Counter
existing_cats = Counter(t["category"] for t in existing_tasks)
print(f"Existing categories:")
for cat, count in existing_cats.items():
    print(f"  {cat}: {count} tasks")

# Validate new tasks
print(f"\nNew tasks to add: {len(NEW_TASKS)}")
new_cats = Counter(t["category"] for t in NEW_TASKS)
print(f"New categories:")
for cat, count in new_cats.items():
    print(f"  {cat}: {count} tasks")

# Combine
all_tasks = existing_tasks + NEW_TASKS
print(f"\nTotal after combining: {len(all_tasks)} tasks")

# Validate total category distribution
print(f"\nFinal category distribution:")
all_cats = Counter(t["category"] for t in all_tasks)
for cat, count in all_cats.items():
    print(f"  {cat}: {count} tasks")

# Save back to tasks.json
with open(TASKS_FILE, "w") as f:
    json.dump(all_tasks, f, indent=2)

print(f"\n✅ Saved {len(all_tasks)} tasks to: {TASKS_FILE}")
print(f"   ({len(existing_tasks)} original + {len(NEW_TASKS)} new)")
print("=" * 60)
print("Ready to run experiment5_adaptevolve.py with 200 tasks.")
print("=" * 60)
