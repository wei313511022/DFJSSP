import json
import time
import gurobipy as gp
from gurobipy import GRB

# Print Gurobi version
try:
    print("Gurobi version:", gp.gurobi.version())
except Exception:
    print("Gurobi version: (unknown)")

# --- GLOBAL PARAMETERS ---
GRID_SIZE = 10
M_BIG = 1000
P_NODES = {1, 6, 10}           # pickup candidate nodes
JSON_STATION_MAPPING = {1: 93, 2: 96, 3: 98}  # station id -> delivery node
M_SET = range(1, 4)            # 3 AGVs
S_m = {1: 1, 2: 6, 3: 10}      # AGV start nodes (grid indices)

def calculate_distance(node1, node2, grid_size=GRID_SIZE):
    """Manhattan distance on GRID_SIZE x GRID_SIZE."""
    r1, c1 = (node1 - 1) // grid_size, (node1 - 1) % grid_size
    r2, c2 = (node2 - 1) // grid_size, (node2 - 1) % grid_size
    return abs(r1 - r2) + abs(c1 - c2)

def load_all_jobs_from_jsonl(filename: str):
    """Read all jobs from a JSONL dispatch file and return a flat list of job dicts."""
    jobs = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                for j in data.get('jobs', []):
                    # ensure fields exist and map station -> g_l
                    station = j.get('station')
                    g_l = JSON_STATION_MAPPING.get(station)
                    if g_l is None:
                        continue
                    jobs.append({
                        'jid': int(j.get('jid')),
                        'proc_time': float(j.get('proc_time', 0.0)),
                        'station': int(station),
                        'g_l': int(g_l),
                        'type': j.get('type', '?')
                    })
    except FileNotFoundError:
        print(f"Error: file not found: {filename}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON decode error: {e}")
        return []
    return jobs

def solve_vrp_from_jobs(jobs):
    """
    Build & solve VRP-like sequencing model following test.py structure.
    Jobs: list of dicts with keys: jid, proc_time (E_l), g_l, type
    Returns model results and computed makespan.
    """
    if not jobs:
        print("No jobs to schedule.")
        return None

    # map tasks to continuous indices L = 1..n
    L_REAL = [j['jid'] for j in jobs]
    n_tasks = len(L_REAL)
    L_SET = range(1, n_tasks + 1)
    VIRTUAL_END = n_tasks + 1
    L_PRIME = range(0, VIRTUAL_END + 1)

    # build TASK_DATA mapping index -> data
    L_REAL_MAP = {idx: L_REAL[idx - 1] for idx in L_SET}
    TASK_DATA = {}
    for idx in L_SET:
        job = next((j for j in jobs if j['jid'] == L_REAL_MAP[idx]), None)
        TASK_DATA[idx] = {'E_l': float(job['proc_time']), 'g_l': int(job['g_l']), 'type': job.get('type', '?')}

    # Model
    model = gp.Model("VRP_from_JSON")
    model.setParam('TimeLimit', 10)
    model.setParam('MIPGap', 0.0)
    model.setParam('OutputFlag', 0)

    # Variables
    Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y_assign")                # assign task->AGV
    W = model.addVars(L_PRIME, L_PRIME, M_SET, vtype=GRB.BINARY, name="W_seq")        # arc variables
    A_P = model.addVars(L_SET, P_NODES, vtype=GRB.BINARY, name="A_pick")              # pick station choice

    T_Pick = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0.0, name="T_pick")
    T_Del = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0.0, name="T_del")
    T_End = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0.0, name="T_end")

    # Objective: same as test.py minimize sum of end times
    model.setObjective(gp.quicksum(T_End[l] for l in L_SET), GRB.MINIMIZE)

    # Constraints
    # each task one pickup station
    model.addConstrs((gp.quicksum(A_P[l, p] for p in P_NODES) == 1 for l in L_SET), name="C1_one_pick")

    # each task assigned to one AGV
    model.addConstrs((gp.quicksum(Y[l, m] for m in M_SET) == 1 for l in L_SET), name="C2_assign")

    # predecessor & successor constraints (each task has one predecessor and one successor over all vehicles)
    for l in L_SET:
        model.addConstr(gp.quicksum(W[lp, l, m] for lp in L_PRIME if lp != l for m in M_SET) == 1, name=f"C3_pred_{l}")
        model.addConstr(gp.quicksum(W[l, ln, m] for ln in L_PRIME if ln != l for m in M_SET) == 1, name=f"C3_succ_{l}")
        # link arcs to Y
        model.addConstrs((W[lp, l, m] <= Y[l, m] for lp in L_PRIME if lp != l for m in M_SET), name=f"C3_link_in_{l}")
        model.addConstrs((W[l, ln, m] <= Y[l, m] for ln in L_PRIME if ln != l for m in M_SET), name=f"C3_link_out_{l}")

    # force each AGV to have exactly one start arc and one end arc (0 -> something and something -> VIRTUAL_END)
    for m in M_SET:
        model.addConstr(gp.quicksum(W[0, l, m] for l in L_SET) == 1, name=f"C4_start_{m}")
        model.addConstr(gp.quicksum(W[l, VIRTUAL_END, m] for l in L_SET) == 1, name=f"C4_end_{m}")

    # disallow arcs back to start
    model.addConstrs((W[l, 0, m] == 0 for l in L_SET for m in M_SET), name="C5_no_return_start")

    # execution time: T_end = T_del + E_l
    model.addConstrs((T_End[l] == T_Del[l] + TASK_DATA[l]['E_l'] for l in L_SET), name="C6_exec")

    # transport time to delivery: T_del >= T_pick + dist(p, g_l) when pickup p selected
    for l in L_SET:
        g_l = TASK_DATA[l]['g_l']
        for p in P_NODES:
            d_pg = calculate_distance(p, g_l)
            model.addConstr(T_Del[l] >= T_Pick[l] + d_pg - M_BIG * (1 - A_P[l, p]), name=f"C7_trans_{l}_{p}")

    # sequencing travel/time gating
    # between tasks: if arc l_prev->l on same AGV and pick p chosen for l then T_pick[l] >= T_end[l_prev] + dist(g_l_prev, p)
    for l in L_SET:
        for lp in L_SET:
            if lp == l:
                continue
            g_lp = TASK_DATA[lp]['g_l']
            for p in P_NODES:
                d_dp = calculate_distance(g_lp, p)
                # use big-M with W[lp,l,m] and A_P[l,p] to activate constraint only when arc and pickup active
                model.addConstrs((T_Pick[l] >= T_End[lp] + d_dp - M_BIG * (2 - W[lp, l, m] - A_P[l, p]) for m in M_SET),
                                 name=f"C8_between_{lp}_{l}_{p}")

        # from start 0: if start arc 0->l for vehicle m and pickup p chosen then T_pick[l] >= dist(S_m, p)
        for m in M_SET:
            Snode = S_m[m]
            for p in P_NODES:
                d_sp = calculate_distance(Snode, p)
                model.addConstr(T_Pick[l] >= d_sp - M_BIG * (2 - W[0, l, m] - A_P[l, p]), name=f"C8_start_{l}_{m}_{p}")

    # solve
    model.update()
    model.optimize()

    # handle infeasible/no-solution
    if model.status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL} or getattr(model, "SolCount", 0) == 0:
        try:
            model.write(f"debug_readjson_{int(time.time())}.lp")
            print("No feasible solution or no solution count. Wrote model for inspection.")
        except Exception:
            print("No feasible solution and failed to write model file.")
        return None

    # extract solution
    W_vals = model.getAttr('X', W)
    A_vals = model.getAttr('X', A_P)
    T_pick_vals = model.getAttr('X', T_Pick)
    T_del_vals = model.getAttr('X', T_Del)
    T_end_vals = model.getAttr('X', T_End)

    # reconstruct per-AGV sequences
    sequence_map = {m: [] for m in M_SET}
    total_transport = 0.0
    for m in M_SET:
        curr = 0
        steps = 0
        seq = []
        while True:
            if steps > n_tasks + 5:
                break
            found = False
            for j in L_PRIME:
                val = W_vals.get((curr, j, m), 0.0)
                if val >= 0.5:
                    # if virtual end reached
                    if j == VIRTUAL_END:
                        curr = j
                        found = True
                        break
                    # j is a task index
                    pick_t = T_pick_vals.get(j, 0.0)
                    del_t = T_del_vals.get(j, 0.0)
                    end_t = T_end_vals.get(j, 0.0)
                    transport = del_t - pick_t
                    proc_time = TASK_DATA[j]['E_l']
                    seq.append({'idx': j, 'jid': L_REAL_MAP[j], 'pick': float(pick_t), 'del': float(del_t),
                                'end': float(end_t), 'transport': float(max(0.0, transport)), 'proc_time': float(proc_time)})
                    total_transport += max(0.0, transport)
                    curr = j
                    found = True
                    break
            if not found:
                break
            if curr == VIRTUAL_END:
                break
            steps += 1
        sequence_map[m] = seq

    # compute makespan = max T_End
    try:
        makespan = max(float(v) for v in T_end_vals.values()) if T_end_vals else 0.0
    except Exception:
        makespan = 0.0

    return {
        'model': model,
        'sequence_map': sequence_map,
        'total_transport_reconstructed': total_transport,
        'makespan': float(makespan),
        'T_end_vals': T_end_vals
    }

def main():
    FILENAME = 'dispatch_inbox_5Jobs_generation.jsonl'
    jobs = load_all_jobs_from_jsonl(FILENAME)
    if not jobs:
        print("No jobs loaded, exiting.")
        return

    print(f"Loaded {len(jobs)} jobs from {FILENAME} (merged all dispatch events).")
    res = solve_vrp_from_jobs(jobs)
    if res is None:
        print("Solver returned no result.")
        return

    seq_map = res['sequence_map']
    print("\n--- Scheduling Result ---")
    for m in M_SET:
        seq = seq_map.get(m, [])
        if not seq:
            print(f"AGV {m}: idle")
            continue
        print(f"AGV {m}: assigned {len(seq)} tasks")
        for t in seq:
            print(f"  jid={t['jid']}, transport={t['transport']:.2f}, proc={t['proc_time']:.2f}, end={t['end']:.2f}")

    print(f"\nTotal reconstructed transport distance (sum del-pick): {res['total_transport_reconstructed']:.2f}")
    print(f"Makespan (max T_end): {res['makespan']:.2f}")

if __name__ == "__main__":
    main()