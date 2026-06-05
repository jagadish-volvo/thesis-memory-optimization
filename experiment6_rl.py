"""
Experiment 6 — Reinforcement Learning-Based Adaptive Coordination
=================================================================
Independent RL module for the orchestration layer of an LLM-based
multi-agent verification workflow at Volvo Group.

PA2534 — Master Thesis in Software Engineering

PROBLEM:
  Static coordination policy always routes task X to agent X.
  Execution tasks = 60% of arrivals → Agent 2 bottleneck.
  Other agents idle while Agent 2 queue grows.

SOLUTION:
  Q-learning RL agent learns to route overflow to idle agents.
  Reduces bottleneck → improves all three thesis KPIs.

KPIs (thesis proposal RQ3):
  1. Overall execution time
  2. Decision latency
  3. Resource utilisation

SIMULATION:
  Tasks arrive faster than one agent can process alone.
  Agent 2 gets overloaded under static policy.
  RL learns to route overflow → distributed load → better KPIs.
"""

import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots_exp6")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_SEED  = 42
NUM_AGENTS   = 5
MAX_QUEUE    = 15
NUM_TRAIN_EP = 10000
NUM_EVAL_EP  = 300
TASKS_PER_EP = 100

# Checkpoints requested by supervisor
CHECKPOINTS = [500, 1000, 2000, 3000, 4000, 5000, 10000]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

AGENT_NAMES  = {0:"Agent 1",1:"Agent 2",2:"Agent 3",
                3:"Agent 4",4:"Agent 5"}

# 60% execution tasks → Agent 2 overloaded under static policy
TASK_WEIGHTS = [5, 10, 60, 15, 10]

# Specialist: 4s, Non-specialist: 7s
# Tasks arrive every 1s → Agent 2 needs 4s per task but gets 1 every 1.67s
# Static: Agent 2 queue grows continuously → high execution time
# RL: distributes to idle agents → queue stays manageable
PROC_TIME = {
    a: {t: 4 if a==t else 7 for t in range(NUM_AGENTS)}
    for a in range(NUM_AGENTS)
}

def new_task():
    return {
        "type":     random.choices(range(NUM_AGENTS), weights=TASK_WEIGHTS)[0],
        "priority": random.choices([0,1,2], weights=[30,50,20])[0]
    }

class WorkflowEnv:
    """
    Simulation where tasks arrive at fixed intervals.
    Each time step = 1 unit of time.
    Agents process tasks over multiple time steps.

    When agent is busy: new task queues up → wait time increases.
    Queue length directly affects execution time and decision latency.

    Static:  Agent 2 always busy → queue grows → high exec time + latency
    RL:      Routes overflow to free agents → queues stay short → lower KPIs
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t             = 0
        self.busy_until    = [0] * NUM_AGENTS
        self.queues        = [[] for _ in range(NUM_AGENTS)]
        self.routed        = [0] * NUM_AGENTS
        self.wait_times    = []
        self.finish_times  = []
        return self._state(new_task())

    def _free(self, a):     return self.busy_until[a] <= self.t
    def _qlen(self, a):     return len(self.queues[a])
    def _remaining(self, a): return max(0, self.busy_until[a] - self.t)

    def _state(self, task):
        qb = [0 if self._qlen(i)==0 else 1 if self._qlen(i)<=3 else 2
              for i in range(NUM_AGENTS)]
        av = [1 if self._free(i) else 0 for i in range(NUM_AGENTS)]
        return (task["type"], task["priority"], *qb, *av)

    def _tick(self):
        """Advance one time step. Complete finished tasks, pull from queues."""
        self.t += 1
        for a in range(NUM_AGENTS):
            if self._free(a) and self.queues[a]:
                next_t = self.queues[a].pop(0)
                pt = PROC_TIME[a][next_t["type"]]
                pt *= 0.8 if next_t["priority"]==2 else 1.2 if next_t["priority"]==0 else 1.0
                self.busy_until[a] = self.t + int(pt)
                self.finish_times.append(self.busy_until[a])

    def step(self, task, action):
        self._tick()
        spec    = task["type"]
        q_len   = self._qlen(action)
        is_free = self._free(action)
        spec_q  = self._qlen(spec)
        idle    = [i for i in range(NUM_AGENTS)
                   if self._free(i) and self._qlen(i)==0]

        # Decision latency = how long this task must wait
        if is_free and q_len == 0:
            wait = 0
        else:
            wait = self._remaining(action) + q_len * int(np.mean(
                [PROC_TIME[action][t] for t in range(NUM_AGENTS)]
            ))
        self.wait_times.append(wait)

        # Reward
        r = 0.0
        if action==spec and is_free and q_len==0:
            r = 2.0
        elif action!=spec and not self._free(spec) and spec_q>=3 and is_free and q_len==0:
            r = 1.5
        elif action==spec and q_len<=2:
            r = 0.8
        elif action!=spec and is_free and q_len==0:
            r = 0.3
        elif q_len>=5 and len(idle)>0:
            r = -1.5
        elif q_len>=MAX_QUEUE-1:
            r = -2.0
        else:
            r = -0.2

        if task["priority"]==2:   r *= 1.4
        elif task["priority"]==0: r *= 0.7
        if len(idle)>=3 and q_len>=3: r -= 0.5

        # Assign
        self.routed[action] += 1
        if is_free and q_len==0:
            pt = PROC_TIME[action][task["type"]]
            pt *= 0.8 if task["priority"]==2 else 1.2 if task["priority"]==0 else 1.0
            self.busy_until[action] = self.t + int(pt)
            self.finish_times.append(self.busy_until[action])
        elif q_len < MAX_QUEUE:
            self.queues[action].append(task)
        else:
            r -= 2.0

        nt = new_task()
        return self._state(nt), nt, r

    def metrics(self):
        exec_t = max(self.finish_times) if self.finish_times else 0
        lat    = np.mean(self.wait_times) if self.wait_times else 0
        total  = sum(self.routed)
        if total == 0:
            util = 0.0
        else:
            ideal = total / NUM_AGENTS
            imb   = sum(abs(self.routed[i]-ideal) for i in range(NUM_AGENTS))
            util  = max(0.0, 1.0 - imb/(2*total))
        return {
            "execution_time":       round(float(exec_t), 1),
            "decision_latency":     round(float(lat), 3),
            "resource_utilisation": round(util, 4),
            "tasks_per_agent":      list(self.routed)
        }


class QLearningAgent:
    def __init__(self, lr=0.15, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=0.9994):
        self.lr, self.gamma = lr, gamma
        self.eps, self.eps_min, self.eps_decay = eps, eps_min, eps_decay
        self.q       = defaultdict(lambda: np.zeros(NUM_AGENTS))
        self.eps_log = []

    def act(self, s, explore=True):
        if explore and random.random() < self.eps:
            return random.randint(0, NUM_AGENTS-1)
        return int(np.argmax(self.q[s]))

    def learn(self, s, a, r, s2):
        td = r + self.gamma*np.max(self.q[s2]) - self.q[s][a]
        self.q[s][a] += self.lr * td

    def decay(self):
        self.eps = max(self.eps_min, self.eps*self.eps_decay)
        self.eps_log.append(self.eps)


class StaticPolicy:
    def act(self, s, explore=False):
        return s[0]


def run(agent, explore=True):
    env   = WorkflowEnv()
    task  = new_task()
    state = env._state(task)
    tot_r = 0.0
    for _ in range(TASKS_PER_EP):
        action      = agent.act(state, explore)
        ns, nt, r   = env.step(task, action)
        if explore and hasattr(agent, 'learn'):
            agent.learn(state, action, r, ns)
        tot_r += r
        state, task  = ns, nt
    return tot_r, env.metrics()


# ── MAIN ──────────────────────────────────────────────────────

print("="*65)
print("EXPERIMENT 6 — RL Adaptive Coordination")
print("Orchestration Layer Optimization")
print("="*65)
print(f"\nAgents:")
for i in range(NUM_AGENTS):
    print(f"  {AGENT_NAMES[i]:12s} | specialist:{PROC_TIME[i][i]}s | non-spec:{PROC_TIME[i][(i+1)%NUM_AGENTS]}s | arrival:{TASK_WEIGHTS[i]/sum(TASK_WEIGHTS)*100:.0f}%")

# Static baseline
print("\n"+"="*65+"CONDITION 1 — STATIC BASELINE")
sp = StaticPolicy()
se,sl,su,srew = [],[],[],[]
for _ in range(NUM_EVAL_EP):
    r,m = run(sp, explore=False)
    se.append(m["execution_time"]); sl.append(m["decision_latency"])
    su.append(m["resource_utilisation"]); srew.append(r)
print(f"\n  KPI 1 Execution time   : {np.mean(se):.1f}s ± {np.std(se):.1f}")
print(f"  KPI 2 Decision latency : {np.mean(sl):.3f}s ± {np.std(sl):.3f}")
print(f"  KPI 3 Resource util    : {np.mean(su):.4f} ± {np.std(su):.4f}")

# RL Training
print("\n"+"="*65)
print(f"TRAINING — {NUM_TRAIN_EP} episodes")
print(f"Checkpoints: {CHECKPOINTS}")
print("="*65)
rl = QLearningAgent()
tr,te,tl,tu = [],[],[],[]

# Checkpoint storage — KPI values at each checkpoint
ckpt_exec  = {}
ckpt_lat   = {}
ckpt_util  = {}

for ep in range(NUM_TRAIN_EP):
    r,m = run(rl, explore=True)
    rl.decay()
    tr.append(r)
    te.append(m["execution_time"])
    tl.append(m["decision_latency"])
    tu.append(m["resource_utilisation"])

    # Record KPI at each checkpoint
    ep1 = ep + 1
    if ep1 in CHECKPOINTS:
        w = min(100, ep1)
        ckpt_exec[ep1]  = round(np.mean(te[-w:]), 3)
        ckpt_lat[ep1]   = round(np.mean(tl[-w:]), 3)
        ckpt_util[ep1]  = round(np.mean(tu[-w:]), 4)
        print(f"  Checkpoint {ep1:6d} | "
              f"Exec:{ckpt_exec[ep1]:6.1f}s | "
              f"Lat:{ckpt_lat[ep1]:.3f}s | "
              f"Util:{ckpt_util[ep1]:.3f} | "
              f"ε:{rl.eps:.3f}")

print(f"\n  ✅ Done | Q-table: {len(rl.q)} states")

# RL Evaluation
print("\n"+"="*65+"CONDITION 2 — RL OPTIMISED")
re,rl2,ru,rrew = [],[],[],[]
for _ in range(NUM_EVAL_EP):
    r,m = run(rl, explore=False)
    re.append(m["execution_time"]); rl2.append(m["decision_latency"])
    ru.append(m["resource_utilisation"]); rrew.append(r)
print(f"\n  KPI 1 Execution time   : {np.mean(re):.1f}s ± {np.std(re):.1f}")
print(f"  KPI 2 Decision latency : {np.mean(rl2):.3f}s ± {np.std(rl2):.3f}")
print(f"  KPI 3 Resource util    : {np.mean(ru):.4f} ± {np.std(ru):.4f}")
print(f"  Q-table states         : {len(rl.q)}")

def pct(b,o,hi=False):
    bm,om=np.mean(b),np.mean(o)
    if bm==0: return 0.0
    return round(((bm-om)/bm*100) if not hi else ((om-bm)/bm*100),1)

ei=pct(se,re); li=pct(sl,rl2); ui=pct(su,ru,hi=True)
print(f"\n  ── IMPROVEMENT ──")
print(f"  KPI 1 Execution time   : {ei:+.1f}%")
print(f"  KPI 2 Decision latency : {li:+.1f}%")
print(f"  KPI 3 Resource util    : {ui:+.1f}%")

# Plots
BLUE,ORANGE,GREEN="#2E5E9E","#E07B35","#2E9E5E"
win=200

fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,(data,lbl,col,sv) in zip(axes,[
    (te,"Execution Time (s)",GREEN,np.mean(se)),
    (tl,"Decision Latency (s)",ORANGE,np.mean(sl)),
    (tu,"Resource Utilisation",GREEN,np.mean(su)),
]):
    sm=np.convolve(data,np.ones(win)/win,mode='valid')
    ax.plot(data,alpha=0.1,color=col,linewidth=0.5)
    ax.plot(range(win-1,len(data)),sm,color=col,linewidth=2,label="RL")
    ax.axhline(sv,color=BLUE,linestyle="--",linewidth=1.5,label=f"Static: {sv:.2f}")
    ax.set_xlabel("Episode",fontsize=10); ax.set_ylabel(lbl,fontsize=10)
    ax.set_title(lbl,fontsize=9,fontweight="bold"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Experiment 6 — RL Training Progress",fontsize=11,fontweight="bold")
plt.tight_layout(); p=os.path.join(PLOTS_DIR,"exp6_training.png"); plt.savefig(p,dpi=150); plt.close(); print(f"\n✅ {p}")

# ── CHECKPOINT PLOT — as requested by supervisor ──────────────
# Shows KPI values at specific episode checkpoints
# X axis: 500, 1000, 2000, 3000, 4000, 5000, 10000
# Y axis: KPI value at that checkpoint
# Answers: at what episode does RL get good + where is the plateau

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

x_points = CHECKPOINTS
exec_vals = [ckpt_exec[c] for c in CHECKPOINTS]
lat_vals  = [ckpt_lat[c]  for c in CHECKPOINTS]
util_vals = [ckpt_util[c] for c in CHECKPOINTS]

static_exec_val = np.mean(se)
static_lat_val  = np.mean(sl)
static_util_val = np.mean(su)

kpi_info = [
    (axes[0], exec_vals, static_exec_val,
     "KPI 1: Execution Time (s)", BLUE,
     "Lower is better"),
    (axes[1], lat_vals,  static_lat_val,
     "KPI 2: Decision Latency (s)", ORANGE,
     "Lower is better"),
    (axes[2], util_vals, static_util_val,
     "KPI 3: Resource Utilisation", GREEN,
     "Higher is better"),
]

for ax, vals, static_val, label, color, direction in kpi_info:
    # Plot KPI values at each checkpoint
    ax.plot(x_points, vals,
            color=color, linewidth=2.5,
            marker="o", markersize=8,
            label=f"RL agent ({direction})")

    # Annotate each checkpoint value
    for x, v in zip(x_points, vals):
        ax.annotate(f"{v:.2f}",
                    xy=(x, v),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=8,
                    color=color, fontweight="bold")

    # Static baseline reference line
    ax.axhline(static_val,
               color="red", linestyle="--",
               linewidth=1.5,
               label=f"Static baseline: {static_val:.3f}")

    ax.set_ylabel(label, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10500)

# X axis labels — exact checkpoints
axes[-1].set_xlabel("Number of Training Episodes", fontsize=12)
axes[-1].set_xticks(CHECKPOINTS)
axes[-1].set_xticklabels([str(c) for c in CHECKPOINTS], fontsize=10)

plt.suptitle(
    "Experiment 6 — KPI Improvement vs Training Episodes\n"
    "Checkpoints: 500, 1000, 2000, 3000, 4000, 5000, 10000",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "exp6_kpi_vs_episodes.png")
plt.savefig(p, dpi=150)
plt.close()
print(f"✅ {p}")

fig,axes=plt.subplots(1,3,figsize=(14,5))
for ax,(title,sv,rv,imp) in zip(axes,[
    ("KPI 1\nExecution Time (s)",se,re,f"{ei:+.1f}%"),
    ("KPI 2\nDecision Latency (s)",sl,rl2,f"{li:+.1f}%"),
    ("KPI 3\nResource Utilisation",su,ru,f"{ui:+.1f}%"),
]):
    ms=[np.mean(sv),np.mean(rv)]; st=[np.std(sv),np.std(rv)]
    bars=ax.bar([0,1],ms,yerr=st,color=[BLUE,ORANGE],alpha=0.85,width=0.5,capsize=6,error_kw={"linewidth":1.5})
    ax.set_xticks([0,1]); ax.set_xticklabels(["Static\nBaseline","RL\nOptimised"],fontsize=10)
    ax.set_title(f"{title}\n{imp}",fontsize=9,fontweight="bold"); ax.grid(axis="y",alpha=0.3)
    for bar,m,s in zip(bars,ms,st):
        ax.text(bar.get_x()+bar.get_width()/2,m+s+max(st)*0.05,f"{m:.3f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
plt.suptitle(f"Experiment 6 — RQ3 ({NUM_EVAL_EP} eval episodes, {TASKS_PER_EP} tasks each)",fontsize=11,fontweight="bold",y=1.02)
plt.tight_layout(); p=os.path.join(PLOTS_DIR,"exp6_kpi_comparison.png"); plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close(); print(f"✅ {p}")

fig,axes=plt.subplots(1,2,figsize=(12,5))
clrs=plt.cm.Blues(np.linspace(0.3,0.9,NUM_AGENTS))
lbls=[f"Agent {i}\n{AGENT_NAMES[i][:5]}" for i in range(NUM_AGENTS)]
_,sm2=run(sp,explore=False); _,rm2=run(rl,explore=False)
for ax,m,title in [(axes[0],sm2,"Static (bottleneck)"),(axes[1],rm2,"RL (balanced)")]:
    bars=ax.bar(lbls,m["tasks_per_agent"],color=clrs,alpha=0.85)
    for bar,cnt in zip(bars,m["tasks_per_agent"]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,str(cnt),ha="center",va="bottom",fontsize=11,fontweight="bold")
    ax.axhline(TASKS_PER_EP/NUM_AGENTS,color="red",linestyle="--",linewidth=1.5,label=f"Ideal:{TASKS_PER_EP/NUM_AGENTS:.0f}")
    ax.set_ylabel("Tasks Routed",fontsize=11); ax.set_title(title,fontsize=10,fontweight="bold")
    ax.set_ylim(0,max(m["tasks_per_agent"])*1.3); ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.3)
plt.suptitle("Task Distribution Across Agents",fontsize=11,fontweight="bold")
plt.tight_layout(); p=os.path.join(PLOTS_DIR,"exp6_load_distribution.png"); plt.savefig(p,dpi=150); plt.close(); print(f"✅ {p}")

fig,ax=plt.subplots(figsize=(9,4))
ax.plot(rl.eps_log,color=ORANGE,linewidth=2)
ax.axhline(rl.eps_min,color="red",linestyle="--",linewidth=1.5,label=f"Min ε={rl.eps_min}")
ax.set_xlabel("Episode",fontsize=11); ax.set_ylabel("Epsilon",fontsize=11)
ax.set_title("Exploration vs Exploitation",fontsize=10,fontweight="bold"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); p=os.path.join(PLOTS_DIR,"exp6_epsilon.png"); plt.savefig(p,dpi=150); plt.close(); print(f"✅ {p}")

results={
    "experiment":"Experiment 6",
    "algorithm":"Q-Learning (tabular, discrete)",
    "config":{
        "num_agents":NUM_AGENTS,
        "training_episodes":NUM_TRAIN_EP,
        "eval_episodes":NUM_EVAL_EP,
        "tasks_per_episode":TASKS_PER_EP,
        "checkpoints":CHECKPOINTS,
        "task_distribution":{AGENT_NAMES[i]:f"{TASK_WEIGHTS[i]/sum(TASK_WEIGHTS)*100:.0f}%" for i in range(NUM_AGENTS)},
        "rl_params":{"lr":rl.lr,"gamma":rl.gamma,"eps_min":rl.eps_min}
    },
    "static_baseline":{
        "execution_time_mean":round(np.mean(se),2),
        "decision_latency_mean":round(np.mean(sl),3),
        "resource_util_mean":round(np.mean(su),4)
    },
    "rl_optimised":{
        "q_table_states":len(rl.q),
        "execution_time_mean":round(np.mean(re),2),
        "decision_latency_mean":round(np.mean(rl2),3),
        "resource_util_mean":round(np.mean(ru),4)
    },
    "rq3_improvements":{
        "execution_time_pct":ei,
        "decision_latency_pct":li,
        "resource_util_pct":ui
    },
    "kpi_checkpoints":{
        "episodes":CHECKPOINTS,
        "execution_time": [ckpt_exec[c] for c in CHECKPOINTS],
        "decision_latency":[ckpt_lat[c]  for c in CHECKPOINTS],
        "resource_util":  [ckpt_util[c] for c in CHECKPOINTS],
        "static_exec":    round(np.mean(se),3),
        "static_lat":     round(np.mean(sl),3),
        "static_util":    round(np.mean(su),4)
    }
}
rp=os.path.join(PLOTS_DIR,"exp6_results.json")
with open(rp,"w") as f: json.dump(results,f,indent=2)
print(f"✅ {rp}")

print("\n"+"="*65)
print("EXPERIMENT 6 COMPLETE")
print("="*65)
print(f"\nRQ3 Results:")
print(f"  {'KPI':<35} {'Static':>10} {'RL':>10} {'Change':>10}")
print(f"  {'-'*67}")
print(f"  {'KPI 1: Execution Time (s)':<35} {np.mean(se):>10.1f} {np.mean(re):>10.1f} {ei:>+9.1f}%")
print(f"  {'KPI 2: Decision Latency (s)':<35} {np.mean(sl):>10.3f} {np.mean(rl2):>10.3f} {li:>+9.1f}%")
print(f"  {'KPI 3: Resource Utilisation':<35} {np.mean(su):>10.4f} {np.mean(ru):>10.4f} {ui:>+9.1f}%")

print(f"\nKPI vs Episodes (checkpoint summary):")
print(f"  {'Episodes':<12} {'Exec (s)':>10} {'Latency (s)':>12} {'Util':>8}")
print(f"  {'Static':<12} {np.mean(se):>10.1f} {np.mean(sl):>12.3f} {np.mean(su):>8.3f}")
print(f"  {'-'*44}")
for c in CHECKPOINTS:
    print(f"  {c:<12} {ckpt_exec[c]:>10.1f} {ckpt_lat[c]:>12.3f} {ckpt_util[c]:>8.3f}")