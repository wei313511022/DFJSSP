import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# --- 1. Environment & Constants Setup ---
AMR_STARTS = {
    "AMR1": (0, 0),
    "AMR2": (0, 5),
    "AMR3": (0, 9),
}
STATIONS = {
    "station1": (9, 2),
    "station2": (9, 5),
    "station3": (9, 7),
}
BASES = list(AMR_STARTS.values())
TYPE_DURATION = {"A": 10, "B": 15, "C": 20}
SUPPLY_LOCATIONS = {"A": AMR_STARTS["AMR1"], "B": AMR_STARTS["AMR2"], "C": AMR_STARTS["AMR3"]}
SCHEDULE_OUTBOX = Path("Random_Job_Arrivals/schedule_outbox.jsonl")
DISPATCH_INBOX = Path("dispatch_inbox_10Jobs_generation.jsonl")
DISPATCH_EVENT_INDEX_ENV = "DISPATCH_EVENT_INDEX"

# Optimization Parameters
JOB_COUNT = 15          # If generating random jobs
POPULATION_SIZE = 60
GENERATIONS = 300       
MUTATION_RATE = 0.2     
STAGNATION_LIMIT = 40   

@dataclass(frozen=True)
class Job:
    idx: int
    type_: str
    duration: float
    station: str

@dataclass
class Individual:
    order: List[int]          # permutation of job indices
    amr_assignment: List[str] # AMR name per job_idx

# --- 2. Helper Functions ---

def grid_distance(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    return float(abs(p[0] - q[0]) + abs(p[1] - q[1]))

def cluster_jobs_by_material(order: List[int], assignments: List[str], jobs: List[Job]) -> List[int]:
    """Sorts jobs to group same AMR and same Material together."""
    keyed = []
    for idx, job_idx in enumerate(order):
        amr = assignments[job_idx]
        job_type = jobs[job_idx].type_
        keyed.append((amr, job_type, idx, job_idx))
    # Sort by AMR, then Material, then original index (stable sort)
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

# --- 3. Simulation Engine (The Core) ---

def get_next_job_for_amr(amr: str, current_pos_in_order: int, order: List[int], assignments: List[str], jobs: List[Job]) -> Optional[Job]:
    """Helper to verify if the AMR has another task later in the sequence."""
    job_map = {job.idx: job for job in jobs}
    for i in range(current_pos_in_order + 1, len(order)):
        next_job_idx = order[i]
        if assignments[next_job_idx] == amr:
            return job_map[next_job_idx]
    return None

def nearest_base_to_station(station: str) -> Tuple[int, int]:
    target = STATIONS[station]
    return min(BASES, key=lambda base: grid_distance(base, target))

def decode_schedule(individual: Individual, jobs: List[Job]) -> Tuple[Dict[str, float], List[Tuple], List[Tuple[int, float]]]:
    """
    Simulates the execution of the schedule.
    Includes inventory check to avoid unnecessary returns to base.
    """
    job_map = {job.idx: job for job in jobs}
    timelines: List[Tuple] = []
    availability = {amr: 0.0 for amr in AMR_STARTS}
    current_position = {amr: AMR_STARTS[amr] for amr in AMR_STARTS}
    inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_STARTS}
    if "AMR1" in inventory: inventory["AMR1"]["A"] = 10
    if "AMR2" in inventory: inventory["AMR2"]["B"] = 10
    if "AMR3" in inventory: inventory["AMR3"]["C"] = 10
    station_available = {station: 0.0 for station in STATIONS}
    order = individual.order
    queue_infos: List[Tuple[int, float]] = []

    for pos, job_idx in enumerate(order):
        job = job_map[job_idx]
        amr = individual.amr_assignment[job_idx]
        material = job.type_
        
        # AMR Start Time
        start_time = availability[amr]
        queue_infos.append((job.idx, start_time))

        # 1. Supply Logic
        if inventory[amr][material] == 0:
            supply_location = SUPPLY_LOCATIONS[material]
            supply_time = int(grid_distance(current_position[amr], supply_location))
            supply_end = start_time + supply_time
            if supply_time > 0:
                timelines.append((amr, start_time, supply_end, "supply", f"Replenish {material}"))
            availability[amr] = supply_end
            current_position[amr] = supply_location
            start_time = supply_end
            inventory[amr][material] = 10 # Refill amount

        # 2. Travel Logic (Directly from current position, which might be the previous station)
        travel_start = availability[amr]
        travel_time = int(grid_distance(current_position[amr], STATIONS[job.station]))
        travel_end = travel_start + travel_time
        if travel_time > 0:
            timelines.append((amr, travel_start, travel_end, "travel", f"Job{job.idx} trans {travel_time}s"))
        
        availability[amr] = travel_end
        current_position[amr] = STATIONS[job.station]
        
        # 3. Wait & Process Logic (Resource Constraint)
        earliest_start = max(travel_end, station_available[job.station])
        if earliest_start > travel_end:
             timelines.append((amr, travel_end, earliest_start, "wait", "Wait Stn"))
        
        process_start = earliest_start
        process_end = process_start + job.duration
        timelines.append((amr, process_start, process_end, f"process_{job.type_}", f"Job{job.idx} {job.type_}({int(job.duration)}s)"))
        
        inventory[amr][material] -= 1
        station_available[job.station] = process_end # Occupy station

        # 4. Look-ahead Return Logic (OPTIMIZED)
        # Check if we have a next job
        next_job = get_next_job_for_amr(amr, pos, order, individual.amr_assignment, jobs)
        
        should_return_to_base = True
        
        if next_job:
            # CORRECTION: If we have inventory for the NEXT job, stay at current station.
            # This allows direct Station-to-Station travel in the next iteration.
            if inventory[amr][next_job.type_] > 0:
                should_return_to_base = False
        else:
            # If no future jobs, we might stay put or return. 
            # Usually end of schedule, doesn't impact makespan unless we care about parking.
            should_return_to_base = False 

        return_start = process_end
        if next_job and should_return_to_base:
            # Only return to base if we really need to (e.g., empty inventory or strategy change)
            # Find base nearest to the NEXT job's station to minimize future travel
            next_base = nearest_base_to_station(next_job.station)
            return_time = int(grid_distance(STATIONS[job.station], next_base))
            return_end = return_start + return_time
            timelines.append((amr, return_start, return_end, "return", f"Return {return_time}s"))
            availability[amr] = return_end
            current_position[amr] = next_base
        else:
            # Stay at current station
            availability[amr] = return_start
            current_position[amr] = STATIONS[job.station]

    return availability, timelines, queue_infos

def fitness(individual: Individual, jobs: List[Job]) -> Tuple[float, List[Tuple]]:
    """
    Score = Makespan + (alpha * TotalActiveTime)
    """
    availability, timeline, _ = decode_schedule(individual, jobs)
    makespan = max(availability.values())
    total_active_time = sum(availability.values())
    weighted_score = makespan + (0.001 * total_active_time)
    return weighted_score, timeline

# --- 4. Genetic Algorithm Operators ---

def random_individual(jobs: List[Job]) -> Individual:
    order = [job.idx for job in jobs]
    random.shuffle(order)
    amr_choices = list(AMR_STARTS.keys())
    assign = [random.choice(amr_choices) for _ in jobs]
    return Individual(order=order, amr_assignment=assign)

def greedy_individual(jobs: List[Job]) -> Individual:
    """Creates an individual with balanced assignment and clustered order."""
    amrs = list(AMR_STARTS.keys())
    assign = []
    for i in range(len(jobs)):
        assign.append(amrs[i % len(amrs)])
    order = [job.idx for job in jobs]
    random.shuffle(order)
    clustered_order = cluster_jobs_by_material(order, assign, jobs)
    return Individual(order=clustered_order, amr_assignment=assign)

def order_crossover(parent_a: Individual, parent_b: Individual, jobs: List[Job]) -> Individual:
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
        if random.random() < 0.5:
            child_assign[idx] = parent_b.amr_assignment[idx]
    clustered_order = cluster_jobs_by_material(child_order, child_assign, jobs)
    return Individual(order=clustered_order, amr_assignment=child_assign)

def smart_load_balance_mutate(individual: Individual, jobs: List[Job]):
    """Moves a job from the busiest AMR to the idlest AMR."""
    availability, _, _ = decode_schedule(individual, jobs)
    busiest_amr = max(availability, key=availability.get)
    idlest_amr = min(availability, key=availability.get)
    if busiest_amr == idlest_amr: return
    busy_job_indices = [i for i, amr in enumerate(individual.amr_assignment) if amr == busiest_amr]
    if busy_job_indices:
        victim_job = random.choice(busy_job_indices)
        individual.amr_assignment[victim_job] = idlest_amr

def mutate(individual: Individual, jobs: List[Job]) -> None:
    size = len(individual.order)
    # 1. Swap
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(size), 2)
        individual.order[i], individual.order[j] = individual.order[j], individual.order[i]
    # 2. Reassign
    for idx in range(size):
        if random.random() < MUTATION_RATE * 0.5: 
            individual.amr_assignment[idx] = random.choice(list(AMR_STARTS.keys()))
    # 3. Cluster Move
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
    # 4. Load Balance
    if random.random() < MUTATION_RATE:
        smart_load_balance_mutate(individual, jobs)

def local_improve(individual: Individual, jobs: List[Job], max_iters: int = 500) -> Individual:
    """Hill Climbing focused on Critical Path."""
    current = Individual(order=list(individual.order), amr_assignment=list(individual.amr_assignment))
    best_score, _ = fitness(current, jobs)
    job_count = len(current.order)
    
    availability, _, _ = decode_schedule(current, jobs)
    critical_amr = max(availability, key=availability.get)
    
    for _ in range(max_iters):
        improved = False
        i, j = random.sample(range(job_count), 2)
        job_i = current.order[i]
        job_j = current.order[j]
        
        if current.amr_assignment[job_i] == critical_amr or current.amr_assignment[job_j] == critical_amr:
            new_order = list(current.order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            neighbor = Individual(order=new_order, amr_assignment=list(current.amr_assignment))
            score, _ = fitness(neighbor, jobs)
            if score < best_score:
                current = neighbor
                best_score = score
                improved = True
                availability, _, _ = decode_schedule(current, jobs)
                critical_amr = max(availability, key=availability.get)

        if not improved:
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

# --- 5. Main Evolution Loop ---

def evolve(jobs: List[Job]) -> Tuple[Individual, List[Tuple]]:
    pop_random_count = int(POPULATION_SIZE * 0.8)
    population = [random_individual(jobs) for _ in range(pop_random_count)]
    population += [greedy_individual(jobs) for _ in range(POPULATION_SIZE - pop_random_count)]
    
    archive_best: Individual = population[0]
    best_fitness = float("inf")
    best_timeline: List[Tuple] = []
    
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
            population = [pair[1] for pair in scored[:5]]
            population += [random_individual(jobs) for _ in range(POPULATION_SIZE - 5)]
            stagnation_counter = 0
            continue 

        new_generation = []
        for _, elite_ind in scored[:2]:
            new_generation.append(Individual(order=list(elite_ind.order), amr_assignment=list(elite_ind.amr_assignment)))

        def get_parent_via_tournament(population_scored, k=3):
            candidates = random.sample(population_scored, k)
            winner = min(candidates, key=lambda x: x[0])
            return winner[1]

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

# --- 6. Visualization & Utilities ---

def plot_gantt(timeline: List[Tuple], queue_infos: List[Tuple[int, float]], jobs: List[Job] = None, solve_time: float = None) -> None:
    """
    Plot Gantt chart using the advanced visual style.
    Fixed: Adjusted x-axis padding so the left sidebar doesn't overlap with the bars.
    """
    # --- Visual Configuration ---
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

    # --- Static Panels ---
    # The sidebar takes up 12% of the total figure width
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

    # --- Plot Limits Calculation (Fixed) ---
    max_timeline_time = max([t[2] for t in timeline]) if timeline else 0.0
    total_job_duration = sum(job.duration for job in jobs) if jobs else 0.0
    
    # Identify the maximum content width needed
    max_plot_time = max(max_timeline_time, total_job_duration)
    
    # Calculate Left Padding
    # Logic: The sidebar takes 12% of the VISUAL width. 
    # We need the negative X range (left padding) to be roughly 15% of the total range 
    # to ensure the sidebar fits entirely in the negative space.
    # Formula: left_pad / (left_pad + right_limit) >= 0.12
    # Using 0.18 (18%) gives us a safe margin.
    LEFT_PAD_RATIO = 0.18 
    
    ax.set_xlim(-max_plot_time * LEFT_PAD_RATIO, max_plot_time * 1.05)
    ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
    ax.set_yticks([])

    # --- Timeline Rectangles ---
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

    # --- Dispatch Queue ---
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

    # --- Title & Final Setup ---
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

# --- 7. Data Loading & Execution ---

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
    availability, decoded_timeline, queue_infos = decode_schedule(individual, jobs)
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
