import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Running on CPU. Install PyTorch with CUDA support.")
    
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import random
import math
import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATASET_PATH': 'training_dataset.jsonl',
    'SAVE_PATH': 'gnn_ddqn_model.pth',
    
    # Physics
    'GRID_WIDTH': 20,
    'SCALE': 5.0,
    'AMR_SPEED': 2.0,
    'CAPACITY_PER_TYPE': 3,
    
    # Training
    'NUM_EPISODES': 1000,
    'BATCH_SIZE': 32,      # Increased for GPU efficiency
    'GAMMA': 0.99,
    'LR': 3e-4,
    'EPS_START': 1.0,
    'EPS_END': 0.05,
    'EPS_DECAY': 1500,
    
    # Model
    'AMR_IN_DIM': 8, 
    'JOB_IN_DIM': 10, 
    'QUEUE_DIM': 3, 
    'HIDDEN_DIM': 64,     # Wider network for GPU
    'ACTION_DIM': 3 
}

# ==========================================
# 2. MAP & PATHFINDING (A*)
# ==========================================
class WarehouseMap:
    def __init__(self):
        self.barriers = set()
        for y in range(5, 15): self.barriers.add((5, y))
        for y in range(5, 15): self.barriers.add((14, y))
        for x in range(8, 12): self.barriers.add((x, 10))

    def get_true_distance(self, start_float, end_float):
        # (Same A* implementation as previous step - condensed for brevity)
        sx, sy = int(start_float[0]/CONFIG['SCALE']), int(start_float[1]/CONFIG['SCALE'])
        ex, ey = int(end_float[0]/CONFIG['SCALE']), int(end_float[1]/CONFIG['SCALE'])
        sx, sy = max(0, min(19, sx)), max(0, min(19, sy))
        ex, ey = max(0, min(19, ex)), max(0, min(19, ey))
        
        start, goal = (sx, sy), (ex, ey)
        if start == goal: return 0.0
        if goal in self.barriers: return 999.0

        frontier = [(0, start)]
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal: break
            x, y = current
            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0<=nx<20 and 0<=ny<20 and (nx,ny) not in self.barriers:
                    new_cost = cost_so_far[current] + 1
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        priority = new_cost + abs(nx-ex) + abs(ny-ey)
                        heapq.heappush(frontier, (priority, (nx, ny)))
                        
        return cost_so_far.get(goal, 999.0) * CONFIG['SCALE']

GLOBAL_MAP = WarehouseMap()

# ==========================================
# 3. ENVIRONMENT
# ==========================================
# (Static Data Generation - Same as before)
def generate_dataset_if_missing():
    if not os.path.exists(CONFIG['DATASET_PATH']):
        print("Generating Data...")
        with open(CONFIG['DATASET_PATH'], 'w') as f:
            for ep in range(CONFIG['NUM_EPISODES']):
                jobs = []
                t = 0
                for i in range(50):
                    t += -math.log(1.0-random.random())*3.0
                    if t>1000: break
                    jobs.append({"id":i, "type":random.choice(["A","B","C"]), 
                               "arrival_time":round(t,2), "dest_station_id":random.randint(1,5)})
                f.write(json.dumps({"episode_id":ep, "jobs":jobs})+"\n")

STATIONS = {
    "SUPPLY_A": (5.0, 95.0), "SUPPLY_B": (5.0, 5.0), "SUPPLY_C": (95.0, 95.0),
    1: (85.0, 85.0), 2: (85.0, 15.0), 3: (50.0, 50.0), 4: (35.0, 15.0), 5: (15.0, 50.0)
}
JOB_PROPS = {"A":10.0, "B":15.0, "C":20.0}
SUPPLY_MAP = {"A":"SUPPLY_A", "B":"SUPPLY_B", "C":"SUPPLY_C"}

@dataclass
class Job:
    jid: int; jtype: str; material: str; arrival_ts: float; proc_time: float
    dest_pos: tuple; supply_pos: tuple; status: int = 0

@dataclass
class AMR:
    aid: int; x: float=10.0; y: float=10.0; status: int=0; remaining_time: float=0.0
    inventory: Dict[str, int] = field(default_factory=lambda:{'A':0,'B':0,'C':0})

class GridEnv:
    def __init__(self):
        self.episodes = []
        with open(CONFIG['DATASET_PATH'], 'r') as f:
            for line in f: self.episodes.append(json.loads(line))
        self.ep_idx = 0

    def reset(self):
        data = self.episodes[self.ep_idx]
        self.ep_idx = (self.ep_idx + 1) % len(self.episodes)
        self.queue = deque()
        for raw in data['jobs']:
            jtype = raw['type']
            self.queue.append(Job(
                jid=raw['id'], jtype=jtype, material=jtype,
                arrival_ts=raw['arrival_time'], proc_time=JOB_PROPS[jtype],
                supply_pos=STATIONS[SUPPLY_MAP[jtype]], dest_pos=STATIONS[raw['dest_station_id']]
            ))
        self.active_jobs = []; self.amrs = [AMR(i) for i in range(3)]; self.sim_time = 0.0
        return self.get_state_arrays() # Return CPU arrays, not GPU tensors

    def assign_nearest(self):
        idle = [a for a in self.amrs if a.status==0]
        jobs = [j for j in self.active_jobs if j.status==1]
        if not idle or not jobs: return
        
        cands = []
        for a in idle:
            for j in jobs:
                mat = j.material
                if a.inventory[mat] == 0 and a.inventory[mat] >= CONFIG['CAPACITY_PER_TYPE']: continue
                
                has_mat = a.inventory[mat] > 0
                dist = GLOBAL_MAP.get_true_distance((a.x,a.y), j.dest_pos if has_mat else j.supply_pos)
                cands.append((dist, a, j, has_mat))
        
        cands.sort(key=lambda x: x[0])
        taken_a, taken_j = set(), set()
        
        for dist, a, j, has_mat in cands:
            if a.aid in taken_a or j.jid in taken_j: continue
            j.status=2; a.status=1
            
            if has_mat:
                a.inventory[j.material] -= 1
                total = dist
            else:
                total = dist + GLOBAL_MAP.get_true_distance(j.supply_pos, j.dest_pos)
            
            a.x, a.y = j.dest_pos
            a.remaining_time = (total/CONFIG['AMR_SPEED']) + j.proc_time
            taken_a.add(a.aid); taken_j.add(j.jid)

    def step(self, action):
        count = {0:0, 1:1, 2:5}[action]
        avail = [j for j in self.queue if j.arrival_ts <= self.sim_time]
        for _ in range(count):
            if avail:
                j = self.queue.popleft(); j.status = 1
                self.active_jobs.append(j); avail.pop(0)
        
        self.assign_nearest()
        self.sim_time += 1.0
        
        for a in self.amrs:
            if a.status==1:
                a.remaining_time -= 1.0
                if a.remaining_time <= 0:
                    a.status=0
                    if self.active_jobs: self.active_jobs.pop(0)

        q_len = len([j for j in self.queue if j.arrival_ts <= self.sim_time])
        shop_load = len(self.active_jobs)
        reward = -0.1*q_len - 0.05*shop_load
        if count>0: reward += 1.0*count
        if shop_load > 12: reward -= 5.0
        
        return self.get_state_arrays(), reward, self.sim_time>=1000

    def get_state_arrays(self):
        # Return Raw Lists (CPU) to save VRAM and Transfer Time
        # AMR: [Status, RemTime, InvA, InvB, InvC, X, Y, 0]
        a_data = [[float(a.status), a.remaining_time/50.0, a.inventory['A']/3.0, 
                   a.inventory['B']/3.0, a.inventory['C']/3.0, a.x/100.0, a.y/100.0, 0.0] for a in self.amrs]
        
        # Job: [1, Proc, Wait, DestX, DestY, SuppX, SuppY, A, B, C]
        j_data = []
        if not self.active_jobs: j_data.append([0.0]*10)
        else:
            for j in self.active_jobs:
                mat = [1,0,0] if j.material=='A' else ([0,1,0] if j.material=='B' else [0,0,1])
                j_data.append([1.0, j.proc_time/20.0, (self.sim_time-j.arrival_ts)/100.0,
                               j.dest_pos[0]/100, j.dest_pos[1]/100, j.supply_pos[0]/100, j.supply_pos[1]/100, *mat])
        
        q_len = len([j for j in self.queue if j.arrival_ts <= self.sim_time])
        q_data = [q_len, 0.0, self.sim_time/1000.0]
        
        return a_data, j_data, q_data

# ==========================================
# 4. BATCHED GNN MODEL (THE GPU FIX)
# ==========================================
class BatchedHeteroGNN(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.upd_amr = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        self.upd_job = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))

    def forward(self, h_amr, h_job, job_mask):
        # h_amr: [Batch, 3, H]
        # h_job: [Batch, MaxJobs, H]
        # job_mask: [Batch, MaxJobs, 1] (1 for real job, 0 for padding)

        # 1. Pool Jobs -> Message to AMR
        # Mask out padding before mean
        masked_job = h_job * job_mask
        # Sum valid jobs and divide by count (avoid div by zero)
        job_sum = masked_job.sum(dim=1, keepdim=True) 
        job_count = job_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        job_mean = job_sum / job_count # [Batch, 1, H]
        
        # Expand to all AMRs
        msg_to_amr = job_mean.expand(-1, h_amr.size(1), -1)
        
        # 2. Pool AMRs -> Message to Job
        amr_mean = h_amr.mean(dim=1, keepdim=True) # [Batch, 1, H]
        msg_to_job = amr_mean.expand(-1, h_job.size(1), -1)

        # 3. Update
        out_amr = self.upd_amr(h_amr + msg_to_amr)
        out_job = self.upd_job(h_job + msg_to_job)
        
        return out_amr, out_job * job_mask # Re-apply mask

class SchedulerAgent(nn.Module):
    def __init__(self):
        super().__init__()
        h = CONFIG['HIDDEN_DIM']
        self.enc_amr = nn.Linear(CONFIG['AMR_IN_DIM'], h)
        self.enc_job = nn.Linear(CONFIG['JOB_IN_DIM'], h)
        self.gnn = BatchedHeteroGNN(h)
        self.head_val = nn.Sequential(nn.Linear(h+CONFIG['QUEUE_DIM'], h), nn.ReLU(), nn.Linear(h, 1))
        self.head_adv = nn.Sequential(nn.Linear(h+CONFIG['QUEUE_DIM'], h), nn.ReLU(), nn.Linear(h, CONFIG['ACTION_DIM']))

    def forward(self, x_amr, x_job, x_q, job_mask):
        # x_amr: [B, 3, 8], x_job: [B, N, 10], mask: [B, N, 1]
        h_amr = F.relu(self.enc_amr(x_amr))
        h_job = F.relu(self.enc_job(x_job))
        
        h_amr, _ = self.gnn(h_amr, h_job, job_mask)
        
        # Global Pooling
        shop_emb = h_amr.mean(dim=1) # [B, H]
        state = torch.cat([shop_emb, x_q], dim=-1)
        
        val = self.head_val(state)
        adv = self.head_adv(state)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# ==========================================
# 5. BATCH PROCESSING UTILS
# ==========================================
def collate_batch(batch_list):
    """
    Takes a list of (amr, job, queue) tuples and stacks them into Tensors.
    Handles variable number of jobs via Padding.
    """
    # Unzip
    states, actions, rewards, next_states, dones = zip(*batch_list)
    
    def pad_and_stack(state_list):
        amrs, jobs, queues = zip(*state_list)
        
        # Stack AMRs (Fixed size 3)
        b_amr = torch.tensor(amrs, dtype=torch.float32, device=CONFIG['DEVICE'])
        b_q = torch.tensor(queues, dtype=torch.float32, device=CONFIG['DEVICE'])
        
        # Pad Jobs (Variable size)
        max_j = max(len(j) for j in jobs)
        b_job = torch.zeros((len(jobs), max_j, CONFIG['JOB_IN_DIM']), dtype=torch.float32, device=CONFIG['DEVICE'])
        b_mask = torch.zeros((len(jobs), max_j, 1), dtype=torch.float32, device=CONFIG['DEVICE'])
        
        for i, j_list in enumerate(jobs):
            L = len(j_list)
            if L > 0:
                tens = torch.tensor(j_list, dtype=torch.float32, device=CONFIG['DEVICE'])
                b_job[i, :L, :] = tens
                b_mask[i, :L, :] = 1.0
                
        return b_amr, b_job, b_q, b_mask

    s_amr, s_job, s_q, s_mask = pad_and_stack(states)
    ns_amr, ns_job, ns_q, ns_mask = pad_and_stack(next_states)
    
    b_a = torch.tensor(actions, device=CONFIG['DEVICE']).unsqueeze(1)
    b_r = torch.tensor(rewards, dtype=torch.float32, device=CONFIG['DEVICE']).unsqueeze(1)
    b_d = torch.tensor(dones, dtype=torch.float32, device=CONFIG['DEVICE']).unsqueeze(1)
    
    return (s_amr, s_job, s_q, s_mask), b_a, b_r, (ns_amr, ns_job, ns_q, ns_mask), b_d

# ==========================================
# 6. TRAINING
# ==========================================
class ReplayBuffer:
    def __init__(self, cap): self.buf = deque(maxlen=cap)
    def push(self, x): self.buf.append(x)
    def sample(self, n): return random.sample(self.buf, n)
    def __len__(self): return len(self.buf)

def optimize(agent, target, opt, memory):
    if len(memory) < CONFIG['BATCH_SIZE']: return 0.0
    
    # 1. Get Batch (CPU Lists)
    batch_raw = memory.sample(CONFIG['BATCH_SIZE'])
    
    # 2. Collate & Move to GPU (ONE operation per batch)
    curr_state, act, rew, next_state, done = collate_batch(batch_raw)
    
    # 3. Forward Pass (Parallel on GPU)
    # Current Q
    q_all = agent(*curr_state)
    q_curr = q_all.gather(1, act)
    
    # Target Q
    with torch.no_grad():
        ns_amr, ns_job, ns_q, ns_mask = next_state
        # Double DQN
        next_acts = agent(ns_amr, ns_job, ns_q, ns_mask).argmax(1, keepdim=True)
        next_vals = target(ns_amr, ns_job, ns_q, ns_mask).gather(1, next_acts)
        q_target = rew + (CONFIG['GAMMA'] * next_vals * (1 - done))
        
    loss = F.smooth_l1_loss(q_curr, q_target)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    opt.step()
    
    return loss.item()

def main():
    print(f"--- GPU STATUS: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ---")
    generate_dataset_if_missing()
    
    env = GridEnv()
    agent = SchedulerAgent().to(CONFIG['DEVICE'])
    target = SchedulerAgent().to(CONFIG['DEVICE'])
    target.load_state_dict(agent.state_dict())
    
    opt = optim.Adam(agent.parameters(), lr=CONFIG['LR'])
    memory = ReplayBuffer(20000)
    
    for ep in range(CONFIG['NUM_EPISODES']):
        state = env.reset() # Returns CPU lists
        ep_rew, ep_loss = 0, 0
        eps = CONFIG['EPS_END'] + (CONFIG['EPS_START'] - CONFIG['EPS_END']) * math.exp(-1.*ep/CONFIG['EPS_DECAY'])
        
        while True:
            # Select Action (Single Inference)
            if random.random() < eps: action = random.randint(0, 2)
            else:
                # Manual Collate for single item
                with torch.no_grad():
                    # Wrap single item in list to reuse collate logic logic-lite
                    # Or just construct tensors quickly
                    s_amr = torch.tensor([state[0]], dtype=torch.float32, device=CONFIG['DEVICE'])
                    s_job = torch.tensor([state[1]], dtype=torch.float32, device=CONFIG['DEVICE'])
                    s_q = torch.tensor([state[2]], dtype=torch.float32, device=CONFIG['DEVICE'])
                    s_mask = torch.ones((1, s_job.size(1), 1), device=CONFIG['DEVICE'])
                    action = agent(s_amr, s_job, s_q, s_mask).argmax(1).item()

            next_state, reward, done = env.step(action)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            ep_rew += reward
            
            if ep > 20: ep_loss += optimize(agent, target, opt, memory)
            if done: break
            
        if ep % 10 == 0:
            target.load_state_dict(agent.state_dict())
            print(f"Ep {ep} | Reward: {ep_rew:.1f} | Loss: {ep_loss:.2f} | Eps: {eps:.2f}")

    torch.save(agent.state_dict(), CONFIG['SAVE_PATH'])
    print("Done.")

if __name__ == "__main__":
    main()