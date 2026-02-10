import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from functools import lru_cache 

# Constants Setup
AMR_STARTS = {
    "AMR1": (0, 7),
    "AMR2": (0, 4),
    "AMR3": (0, 1),
}
STATIONS = {
    "station1": (9, 8),
    "station2": (9, 6),
    "station3": (9, 4),
    "station4": (9, 2),
    "station5": (9, 0),
}
OBSTACLES = {
    (6,8),(6,9),(6,6),(6,5),(6,4),(6,2),(6,1),(6,0)
}
_GRID_POINTS = list(AMR_STARTS.values()) + list(STATIONS.values()) + list(OBSTACLES)
GRID_MIN_X = min(p[0] for p in _GRID_POINTS)
GRID_MAX_X = max(p[0] for p in _GRID_POINTS)
GRID_MIN_Y = min(p[1] for p in _GRID_POINTS)
GRID_MAX_Y = max(p[1] for p in _GRID_POINTS)
BASES = list(AMR_STARTS.values())
TYPE_DURATION = {"A": 10, "B": 15, "C": 20}
SUPPLY_LOCATIONS = {"A": AMR_STARTS["AMR1"], "B": AMR_STARTS["AMR2"], "C": AMR_STARTS["AMR3"]}
SCHEDULE_OUTBOX = Path("Random_Job_Arrivals/schedule_outbox.jsonl")
DISPATCH_INBOX = Path("dispatch_inbox_10Jobs_generation.jsonl")
DISPATCH_EVENT_INDEX_ENV = "DISPATCH_EVENT_INDEX"

JOB_COUNT = 25        
POPULATION_SIZE = 60    # number of candidate solutions
GENERATIONS = 300       
MUTATION_RATE = 0.2     
STAGNATION_LIMIT = 40   # number of convergence iterations 

@dataclass(frozen=True)
class Job:
    idx: int
    type_: str
    duration: float
    station: str

# one solution in GA
@dataclass
class Individual:
    order: List[int]          # permutation of job execution order
    amr_assignment: List[str] # job assigned to amr

# check whether routing within the bound
def _is_within_bounds(point: Tuple[int, int]) -> bool:
    x, y = point
    return GRID_MIN_X <= x <= GRID_MAX_X and GRID_MIN_Y <= y <= GRID_MAX_Y

def _adjacent_points(point: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = point
    # right , left , up , down 
    deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    neighbors = []
    for dx, dy in deltas:
        candidate = (x + dx, y + dy)
        # check whether routing within the bound
        if not _is_within_bounds(candidate):
            continue
        # check whether collision with the obstacles
        if candidate in OBSTACLES:
            continue
        neighbors.append(candidate)
    return neighbors #return legal can move adjacent_points

#Tracing back the complete path from parents
def _build_path(parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents.get(current)
    return list(reversed(path))


def _manhattan_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [start]
    x, y = start
    tx, ty = end
    dx = 1 if tx > x else -1
    while x != tx:
        x += dx
        path.append((x, y))
    dy = 1 if ty > y else -1
    while y != ty:
        y += dy
        path.append((x, y))
    return path

# Use BFS to find the shortest path between start and end
# In this way, when the program queries the same start and end points again, it will not run BFS again, but will directly return the previously calculated path.
@lru_cache(maxsize=None)
def shortest_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    if start == end:
        return [start]
    parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    queue = deque([start])
    visited = {start}
    while queue:
        current = queue.popleft()
        for neighbor in _adjacent_points(current):
            if neighbor in visited:
                continue
            parents[neighbor] = current
            if neighbor == end:
                return _build_path(parents, end) #backtrack the path
            visited.add(neighbor)
            queue.append(neighbor)
    return _manhattan_path(start, end)

# Help a certain AMR accumulate the paths it has traveled.
def _extend_path_log(path_logs: Dict[str, List[Tuple[int, int]]], amr: str, segment: List[Tuple[int, int]]) -> None:
    if len(segment) <= 1:
        return
    log = path_logs[amr]
    if log and log[-1] == segment[0]:
        log.extend(segment[1:])
    else:
        log.extend(segment)

# The actual path distance between two points
def grid_distance(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    # cahce's shortest_path
    path = shortest_path(p, q)
    return float(len(path) - 1)

# Reorder the job order so that Jobs on the same AMR and of the same job type are grouped together as much as possible
def cluster_jobs_by_material(order: List[int], assignments: List[str], jobs: List[Job]) -> List[int]:
    keyed = []
    for idx, job_idx in enumerate(order):
        amr = assignments[job_idx]
        job_type = jobs[job_idx].type_
        keyed.append((amr, job_type, idx, job_idx))
    # Sort by AMR, then Material, then original index 
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in keyed]

def find_adjacent_blocks(order: List[int], assignments: List[str], jobs: List[Job]) -> List[Tuple[int, int]]:
    """Identifies blocks of consecutive jobs with same AMR and Type."""
    job_count = len(order)
    blocks = []
    idx = 0
    while idx < job_count:
        start = idx
        current_job = order[idx]
        curr_amr = assignments[current_job]
        curr_type = jobs[current_job].type_
        idx += 1
        while idx < job_count:
            next_job = order[idx]
            if assignments[next_job] == curr_amr and jobs[next_job].type_ == curr_type:
                idx += 1
            else:
                break
        if idx - start > 1:
            blocks.append((start, idx))
    return blocks

# Verify if the AMR has another task later in the sequence.If have,return the task,otherwise return null.
def get_next_job_for_amr(amr: str, current_pos_in_order: int, order: List[int], assignments: List[str], jobs: List[Job]) -> Optional[Job]:
    job_map = {job.idx: job for job in jobs}
    for i in range(current_pos_in_order + 1, len(order)):
        next_job_idx = order[i]
        if assignments[next_job_idx] == amr:
            return job_map[next_job_idx]
    return None

# Find the AMR_station closest to a certain workstation
def nearest_base_to_station(station: str) -> Tuple[int, int]:
    target = STATIONS[station]
    return min(BASES, key=lambda base: grid_distance(base, target))

def decode_schedule(individual: Individual, jobs: List[Job], need_log: bool = False) -> Tuple[Dict[str, float], List[Tuple], List[Tuple[int, float]], Dict[str, List[Tuple[int, int]]]]:
    job_map = {job.idx: job for job in jobs} # get job information
    timelines: List[Tuple] = []
    availability = {amr: 0.0 for amr in AMR_STARTS} # amr availability time
    current_position = {amr: AMR_STARTS[amr] for amr in AMR_STARTS} # current position of each AMR
    path_logs = {amr: [current_position[amr]] for amr in AMR_STARTS} if need_log else {}
    inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_STARTS} # How much material do we currently have on hand for this AMR?
    if "AMR1" in inventory: inventory["AMR1"]["A"] = 3
    if "AMR2" in inventory: inventory["AMR2"]["B"] = 3
    if "AMR3" in inventory: inventory["AMR3"]["C"] = 3
    station_available = {station: 0.0 for station in STATIONS} # station availability time
    order = individual.order
    queue_infos: List[Tuple[int, float]] = []

    for pos, job_idx in enumerate(order):
        job = job_map[job_idx]
        amr = individual.amr_assignment[job_idx]
        material = job.type_
        # AMR Start Time
        start_time = availability[amr]
        queue_infos.append((job.idx, start_time))
        # fill material
        if inventory[amr][material] == 0:
            supply_location = SUPPLY_LOCATIONS[material]
            supply_path = shortest_path(current_position[amr], supply_location)
            supply_time = int(len(supply_path) - 1)
            supply_end = start_time + supply_time
            if supply_time > 0:
                timelines.append((amr, start_time, supply_end, "supply", f"Replenish {material}"))
            availability[amr] = supply_end
            current_position[amr] = supply_location
            start_time = supply_end
            inventory[amr][material] = 3 # Refill amount
            if need_log: _extend_path_log(path_logs, amr, supply_path)

        # From your current location, navigate to job.station
        travel_start = availability[amr]
        travel_path = shortest_path(current_position[amr], STATIONS[job.station])
        travel_time = int(len(travel_path) - 1)
        travel_end = travel_start + travel_time
        if travel_time > 0:
            timelines.append((amr, travel_start, travel_end, "travel", f"Job{job.idx} trans {travel_time}s"))
        availability[amr] = travel_end
        current_position[amr] = STATIONS[job.station]
        if need_log: _extend_path_log(path_logs, amr, travel_path)

        # Wait station availability if needed
        earliest_start = max(travel_end, station_available[job.station])
        if earliest_start > travel_end:
             timelines.append((amr, travel_end, earliest_start, "wait", "Wait Stn"))
        process_start = earliest_start
        process_end = process_start + job.duration
        timelines.append((amr, process_start, process_end, f"process_{job.type_}", f"Job{job.idx} {job.type_}({int(job.duration)}s)"))
        inventory[amr][material] -= 1
        station_available[job.station] = process_end # Occupy station
        # Look-ahead Return Logic (OPTIMIZED)
        # Check if we have a next job
        next_job = get_next_job_for_amr(amr, pos, order, individual.amr_assignment, jobs)
        should_return_to_base = True
        if next_job:
            if inventory[amr][next_job.type_] > 0: # have inventory for next job
                should_return_to_base = False
        else:
            # If no future jobs, we might stay at workstation instead of returning to base.
            should_return_to_base = False 

        return_start = process_end
        if next_job and should_return_to_base:
        # If you're going back to base, go back to the base that's "most advantageous for your next job."
            next_base = nearest_base_to_station(next_job.station)
            return_path = shortest_path(STATIONS[job.station], next_base)
            return_time = int(len(return_path) - 1)
            return_end = return_start + return_time
            timelines.append((amr, return_start, return_end, "return", f"Return {return_time}s"))
            availability[amr] = return_end
            current_position[amr] = next_base
            if need_log: _extend_path_log(path_logs, amr, return_path)
        else:
            # Stay at current station
            availability[amr] = return_start
            current_position[amr] = STATIONS[job.station]

    return availability, timelines, queue_infos, path_logs

def fitness(individual: Individual, jobs: List[Job]) -> Tuple[float, List[Tuple]]:
    availability, timeline, _, _ = decode_schedule(individual, jobs, need_log=False)
    makespan = max(availability.values())
    #sum of all AMR finish time
    total_active_time = sum(availability.values())
    # Fitness=Makespan + (alpha * TotalActiveTime) ,alpha=0.001
    # Primary consider the makespan, secondary consider the total load balance 
    weighted_score = makespan + (0.001 * total_active_time)
    return weighted_score, timeline

# Random individual generation
# Can increase diversity of initial population
def random_individual(jobs: List[Job]) -> Individual:
    order = [job.idx for job in jobs]
    random.shuffle(order)
    amr_choices = list(AMR_STARTS.keys())
    assign = [random.choice(amr_choices) for _ in jobs]
    return Individual(order=order, amr_assignment=assign)

# Creates an individual with balanced assignment and clustered order.
# This will guide the GA search space toward a certain type of solution, causing premature convergence to a suboptimal solution.
def greedy_individual(jobs: List[Job]) -> Individual:
    amrs = list(AMR_STARTS.keys())
    assign = []
    # Assign using round-robin: job0→AMR1, job1→AMR2, job2→AMR3, job3→AMR1… .It's about load balancing from the start.
    for i in range(len(jobs)):
        assign.append(amrs[i % len(amrs)])
    # First, shuffle the orders, then use cluster_jobs_by_material() to group jobs with the same AMR and type together as much as possible.
    order = [job.idx for job in jobs]
    random.shuffle(order)
    clustered_order = cluster_jobs_by_material(order, assign, jobs)
    return Individual(order=clustered_order, amr_assignment=assign)

def order_crossover(parent_a: Individual, parent_b: Individual, jobs: List[Job]) -> Individual:
    # Take a random segment from parent A [a:b).Place this segment unchanged into the child's position.Fill in the remaining positions in the order of parent B.
    size = len(parent_a.order)
    a, b = sorted(random.sample(range(size), 2))
    child_order = [-1] * size
    child_order[a:b] = parent_a.order[a:b]
    fill = [gene for gene in parent_b.order if gene not in child_order]
    ptr = 0
    for i in range(size):
        if child_order[i] == -1:
            child_order[i] = fill[ptr]
            ptr += 1
    child_assign = parent_a.amr_assignment[:]
    for idx in range(size):
        # 50% chance to inherit AMR assignment from parent B or parent A
        if random.random() < 0.5:
            child_assign[idx] = parent_b.amr_assignment[idx]
    # Reorder the job order so that Jobs on the same AMR and of the same job type are grouped together as much as possible 
    clustered_order = cluster_jobs_by_material(child_order, child_assign, jobs)
    return Individual(order=clustered_order, amr_assignment=child_assign)

# Moves a job from the busiest AMR to the idlest AMR.
def smart_load_balance_mutate(individual: Individual, jobs: List[Job]):
    availability, _, _, _ = decode_schedule(individual, jobs, need_log=False) 
    busiest_amr = max(availability, key=availability.get) # find the busiest amr
    idlest_amr = min(availability, key=availability.get)  # find the idlest amr
    if busiest_amr == idlest_amr: return
    busy_job_indices = [i for i, amr in enumerate(individual.amr_assignment) if amr == busiest_amr]
    # Randomly select a job from busiest_amr. Reassign that job directly to idlest_amr.
    if busy_job_indices:
        victim_job = random.choice(busy_job_indices)
        individual.amr_assignment[victim_job] = idlest_amr

def mutate(individual: Individual, jobs: List[Job]) -> None:
    size = len(individual.order)
    # Swap the positions of the two jobs
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(size), 2)
        individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
    # Randomly modify the AMR of certain jobs
    for idx in range(size):
        if random.random() < MUTATION_RATE * 0.5: 
            individual.amr_assignment[idx] = random.choice(list(AMR_STARTS.keys()))
    # Move a specific job next to a job with the same AMR and the same type.
    if random.random() < MUTATION_RATE:
        idx = random.randrange(size)
        job_idx = individual.order[idx]
        target_type = jobs[job_idx].type_
        target_amr = individual.amr_assignment[job_idx]
        for target_idx, other_job in enumerate(individual.order):
            if target_idx == idx: continue
            if (individual.amr_assignment[other_job] == target_amr and jobs[other_job].type_ == target_type):
                individual.order.pop(idx)
                insert_idx = target_idx if target_idx < idx else target_idx
                individual.order.insert(insert_idx + (1 if target_idx >= idx else 0), job_idx)
                break
    # Load Balance
    if random.random() < MUTATION_RATE:
        smart_load_balance_mutate(individual, jobs)

def local_improve(individual: Individual, jobs: List[Job], max_iters: int = 500) -> Individual:
    current = Individual(order=list(individual.order), amr_assignment=list(individual.amr_assignment))
    best_score, _ = fitness(current, jobs) # Current best fitness
    job_count = len(current.order)
    availability, _, _, _ = decode_schedule(current, jobs, need_log=False)
    critical_amr = max(availability, key=availability.get) # AMR with the longest makespan
    for _ in range(max_iters):
        # Each time, randomly select two job_i and job_j and try to swap them.One of them needs to belong to the critical AMR.
        improved = False
        i, j = random.sample(range(job_count), 2)
        job_i = current.order[i]
        job_j = current.order[j]
        if current.amr_assignment[job_i] == critical_amr or current.amr_assignment[job_j] == critical_amr:
            new_order = list(current.order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
            score, _ = fitness(neighbor, jobs)
            # Only accept the neighbor if it improves the fitness
            if score < best_score:
                current = neighbor
                best_score = score
                improved = True
                availability, _, _, _ = decode_schedule(current, jobs, need_log=False)
                critical_amr = max(availability, key=availability.get)
        if not improved:
            # Find consecutive blocks with the same AMR and type in the current order.Randomly select one block.Move the entire block to another location.
            blocks = find_adjacent_blocks(current.order, current.amr_assignment, jobs)
            if blocks:
                start, end = random.choice(blocks)
                block = current.order[start:end]
                remainder = current.order[:start] + current.order[end:]
                insert_pos = random.randint(0, len(remainder))
                new_order = remainder[:insert_pos] + block + remainder[insert_pos:]
                neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
                score, _ = fitness(neighbor, jobs)
                if score < best_score:
                    current = neighbor
                    best_score = score
                    improved = True
        if improved: continue
    return current

def evolve(jobs: List[Job]) -> Tuple[Individual, List[Tuple]]:
    # Initial Population = 80% random group + 20% greedy group
    pop_random_count = int(POPULATION_SIZE * 0.8)
    population = [random_individual(jobs) for _ in range(pop_random_count)]
    population += [greedy_individual(jobs) for _ in range(POPULATION_SIZE - pop_random_count)]
    # Global best solution
    archive_best: Individual = population[0]
    best_fitness = float("inf")
    best_timeline: List[Tuple] = []
    # counter the number of generations without improvement
    stagnation_counter = 0
    for gen in range(GENERATIONS):
        scored = []
        for ind in population:
            m, _ = fitness(ind, jobs)
            scored.append((m, ind))
        scored.sort(key=lambda pair: pair[0])
        current_best = scored[0][1]
        f_val = scored[0][0]
        if f_val < best_fitness:
            best_fitness = f_val
            best_timeline = fitness(current_best, jobs)[1]
            archive_best = current_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        if stagnation_counter > STAGNATION_LIMIT:
            # Keep the top 5 elites.Re-randomize all others.
            population = [pair[1] for pair in scored[:5]]
            population += [random_individual(jobs) for _ in range(POPULATION_SIZE - 5)]
            stagnation_counter = 0
            continue 
        # Elitism: Top 2 students directly admitted to the next generation.
        new_generation = []
        for _, elite_ind in scored[:2]:
            new_generation.append(Individual(order=list(elite_ind.order), amr_assignment=list(elite_ind.amr_assignment)))
        # Randomly select k people.Choose the one with the best fitness to be a parent.
        def get_parent_via_tournament(population_scored, k=3):
            candidates = random.sample(population_scored, k)
            winner = min(candidates, key=lambda x: x[0])
            return winner[1]
        # Crossover+mutate
        while len(new_generation) < POPULATION_SIZE:
            parent_a = get_parent_via_tournament(scored, k=3)
            parent_b = get_parent_via_tournament(scored, k=3)
            child = order_crossover(parent_a, parent_b, jobs)
            mutate(child, jobs)
            new_generation.append(child)

        population = new_generation

    archive_best = local_improve(archive_best, jobs)
    makespan, timeline = fitness(archive_best, jobs)
    return archive_best, timeline


def plot_gantt(timeline: List[Tuple], queue_infos: List[Tuple[int, float]], jobs: List[Job] = None, solve_time: float = None) -> None:
    AMR_COUNT = len(AMR_STARTS)
    AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
    BOTTOM_MIN = 0.0
    BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0 
    AMR_Y_CENTERS = [BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT) for i in range(AMR_COUNT)]
    AMR_LANE_H = BOTTOM_HEIGHT / AMR_COUNT * 0.7
    _cycle = plt.rcParams.get("axes.prop_cycle", None)
    TYPE_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    TRANSPORT_COLOR = "lightgray"
    WAIT_COLOR = "lightgray"
    sorted_amrs = sorted(AMR_STARTS.keys())
    amr_y_map = {name: AMR_Y_CENTERS[i] for i, name in enumerate(sorted_amrs)}
    fig, ax = plt.subplots(figsize=(14, 6))
    SIDEBAR_WIDTH_FRAC = 0.12 
    def draw_static_panels(ax):
        top_panel = Rectangle((0.0, 0.5), SIDEBAR_WIDTH_FRAC, 0.5, transform=ax.transAxes,
                              fill=False, linewidth=1.5, edgecolor="black", clip_on=False, zorder=10)
        ax.add_patch(top_panel)
        tp_text = ax.text(SIDEBAR_WIDTH_FRAC * 0.5, 0.75, "Dispatching\nQueue", transform=ax.transAxes,
                          ha="center", va="center", fontsize=10, weight="bold", color="gray", zorder=11)
        tp_text.set_clip_path(top_panel)

        bot_panel = Rectangle((0.0, 0.0), SIDEBAR_WIDTH_FRAC, 0.5, transform=ax.transAxes,
                              fill=False, linewidth=1.5, edgecolor="black", clip_on=False, zorder=10)
        ax.add_patch(bot_panel)
        for i, name in enumerate(sorted_amrs):
            y_frac = (i + 0.5) / AMR_COUNT * 0.5
            txt = ax.text(SIDEBAR_WIDTH_FRAC * 0.5, y_frac, name, transform=ax.transAxes,
                          ha="center", va="center", fontsize=10, weight="bold", color="gray", zorder=11)
            txt.set_clip_path(bot_panel)
    draw_static_panels(ax)
    max_timeline_time = max([t[2] for t in timeline]) if timeline else 0.0
    total_job_duration = sum(job.duration for job in jobs) if jobs else 0.0
    max_plot_time = max(max_timeline_time, total_job_duration)
    LEFT_PAD_RATIO = 0.18 
    ax.set_xlim(-max_plot_time * LEFT_PAD_RATIO, max_plot_time * 1.05)
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_yticks([])
    for entry in timeline:
        amr, start, end, kind, label = entry
        duration = end - start
        if duration <= 0: continue
        y_c = amr_y_map.get(amr, 0)
        rect_kwargs = {"linewidth": 0.5, "edgecolor": "gray", "alpha": 1.0, "clip_on": True}
        text_color = "black"
        font_weight = "bold"
        font_size = 7
        display_label = label
        if kind.startswith("process"):
            jtype = kind.split("_")[-1]
            rect_kwargs["facecolor"] = TYPE_COLORS.get(jtype, "gray")
            rect_kwargs["edgecolor"] = "black"
            rect_kwargs["linewidth"] = 1.2
            rect_kwargs["zorder"] = 5
            text_color = "white"
            try:
                parts = label.split(" ")
                jid = parts[0].replace("Job", "J_")
                dur = parts[1].split("(")[1].replace("s)", "")
                display_label = f"{jid}\n({dur})"
            except:
                pass
        elif kind in ["travel", "return", "supply"]:
            rect_kwargs["facecolor"] = TRANSPORT_COLOR
            rect_kwargs["edgecolor"] = "gray"
            rect_kwargs["hatch"] = "///"
            rect_kwargs["zorder"] = 2
            font_weight = "normal"
            if "trans" in label:
                dur = label.split(" ")[-1].replace("s", "")
                display_label = f"({dur})"
            elif "Return" in label:
                dur = label.split(" ")[-1].replace("s", "")
                display_label = f"Ret\n({dur})"
            elif "Replenish" in label:
                display_label = "Supply"
        else:
            rect_kwargs["facecolor"] = WAIT_COLOR
            rect_kwargs["hatch"] = ".."
            display_label = "W"
        r = Rectangle((start, y_c - AMR_LANE_H / 2), duration, AMR_LANE_H, **rect_kwargs)
        ax.add_patch(r)
        ax.text(start + duration / 2, y_c, display_label, ha="center", va="center", 
                fontsize=font_size, color=text_color, weight=font_weight, zorder=6)
    if queue_infos and jobs:
        queue_y_center = 1.25
        queue_h = 0.5
        job_map = {job.idx: job for job in jobs}
        sorted_queue = sorted(queue_infos, key=lambda entry: entry[0])
        current_x = 0.0
        for job_idx, _ in sorted_queue:
            job = job_map.get(job_idx)
            if not job: continue
            width = job.duration
            face = TYPE_COLORS.get(job.type_, "gray")
            r = Rectangle((current_x, queue_y_center - queue_h / 2), width, queue_h,
                          facecolor=face, edgecolor="black", linewidth=1.2, zorder=5)
            ax.add_patch(r)
            ax.text(current_x + width / 2, queue_y_center, f"J_{job_idx}", 
                    ha="center", va="center", color="white", fontsize=9, weight="bold", zorder=6)
            current_x += width
    ax.set_xlabel("Time (s)", fontweight="bold")
    title_text = f"AMR Schedule (Optimized GA) | Makespan: {max_timeline_time:.1f}s"
    if solve_time is not None:
        title_text += f" | Solve: {solve_time:.4f}s"
    ax.set_title(title_text, pad=10)
    ax.grid(True, axis="x", linestyle=":", color='gray', alpha=0.5, zorder=0)
    handles = [Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}") for k in sorted(TYPE_COLORS.keys())]
    handles.append(Patch(facecolor=TRANSPORT_COLOR, edgecolor="gray", hatch="///", label="Transportation"))
    handles.append(Patch(facecolor=WAIT_COLOR, edgecolor="gray", hatch="..", label="Waiting"))
    ax.legend(handles=handles, loc="upper right", frameon=True, bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.show()
# Data Loading & Execution 
def station_key_from_value(raw_station: Optional[str]) -> Optional[str]:
    if raw_station is None: return None
    try: station_id = int(raw_station)
    except: return None
    key = f"station{station_id}"
    return key if key in STATIONS else None
def load_dispatch_events(path: Path = DISPATCH_INBOX) -> List[Dict[str, object]]:
    events = []
    if not path.exists(): return events
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for idx, payload in enumerate(lines):
        try: data = json.loads(payload)
        except: continue
        jobs = []
        for job_idx, raw_job in enumerate(data.get("jobs", [])):
            station_key = station_key_from_value(raw_job.get("station"))
            if station_key is None: continue
            type_ = str(raw_job.get("type", "")).upper()
            if type_ not in TYPE_DURATION: type_ = "A"
            jobs.append(Job(idx=job_idx, type_=type_, duration=float(raw_job.get("proc_time", TYPE_DURATION[type_])), station=station_key))
        if jobs:
            events.append({"index": idx, "dispatch_time": float(data.get("dispatch_time", 0.0)), "jobs": jobs})
    return events
def make_jobs() -> List[Job]:
    stations = list(STATIONS.keys())
    jobs = []
    for idx in range(JOB_COUNT):
        type_ = random.choice(list(TYPE_DURATION.keys()))
        station = random.choice(stations)
        jobs.append(Job(idx=idx, type_=type_, duration=TYPE_DURATION[type_], station=station))
    return jobs
def describe_solution(individual: Individual, jobs: List[Job], timeline: List[Tuple], solve_time: float = None) -> None:
    availability, decoded_timeline, queue_infos, path_logs = decode_schedule(individual, jobs, need_log=True)
    makespan = max(availability.values())
    print(f"Optimal Makespan Found: {makespan:.2f}s")
    plot_gantt(decoded_timeline, queue_infos, jobs, solve_time=solve_time)
if __name__ == "__main__":
    random.seed(42)
    dispatch_events = load_dispatch_events()
    target_index = os.environ.get(DISPATCH_EVENT_INDEX_ENV)
    
    if dispatch_events:
        if target_index is not None:
             dispatch_events = [e for e in dispatch_events if str(e["index"]) == str(target_index)]
        
        for event in dispatch_events:
            print(f"\n=== Processing Dispatch Event {event['index']} (Jobs: {len(event['jobs'])}) ===")
            start_time = time.perf_counter()
            best_ind, timeline = evolve(event["jobs"])
            solve_dur = time.perf_counter() - start_time
            describe_solution(best_ind, event["jobs"], timeline, solve_time=solve_dur)
    else:
        print("No dispatch file found. Generating random jobs...")
        jobs = make_jobs()
        start_time = time.perf_counter()
        best_ind, timeline = evolve(jobs)
        solve_dur = time.perf_counter() - start_time
        describe_solution(best_ind, jobs, timeline, solve_time=solve_dur)
