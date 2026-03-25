from typing import Dict, List, Tuple
from collections import deque
import heapq
import collections
from GA import (
    Job, Individual, AMR_STARTS, AMR_KEYS, STATIONS, OBSTACLES, _GRID_POINTS,
    GRID_MIN_X, GRID_MAX_X, GRID_MIN_Y, GRID_MAX_Y, BASES, TYPE_DURATION,
    SUPPLY_LOCATIONS, MAX_DEPTH, _is_within_bounds, _DELTAS, _adjacent_points,
    _build_path, _manhattan_path, heuristic, shortest_path, _extend_path_log, grid_distance,
    nearest_base_to_station, get_next_job_for_amr
)

def decode_schedule_tick_by_tick(individual: Individual, jobs: List[Job], need_log: bool = False, check_collision: bool = True):
    if not check_collision:
        from GA import decode_schedule
        return decode_schedule(individual, jobs, need_log, False)

    job_map = {job.idx: job for job in jobs}
    amr_queues = {amr: deque() for amr in AMR_STARTS}
    for job_idx, amr in zip(individual.order, individual.amr_assignment):
        amr_queues[amr].append(job_map[job_idx])
        
    positions = {amr: AMR_STARTS[amr] for amr in AMR_STARTS}
    inventory = {amr: {mat: 0 for mat in TYPE_DURATION.keys()} for amr in AMR_STARTS}
    if "AMR1" in inventory: inventory["AMR1"]["A"] = 3
    if "AMR2" in inventory: inventory["AMR2"]["B"] = 3
    if "AMR3" in inventory: inventory["AMR3"]["C"] = 3
    
    amr_states = {amr: {'mode': 'idle', 'goal': None, 'job': None, 'proc_ticks': 0} for amr in AMR_STARTS}
    path_logs = {amr: [positions[amr]] for amr in AMR_STARTS} if need_log else {}
    timelines: List[Tuple] = []
    queue_infos: List[Tuple[int, float]] = []
    invalid_jobs_count = 0
    
    station_occupied = {s: False for s in STATIONS}
    
    t = 0
    while True:
        all_idle_and_empty = True
        for amr in AMR_KEYS:
            if len(amr_queues[amr]) > 0 or amr_states[amr]['mode'] != 'idle':
                all_idle_and_empty = False
                break
        if all_idle_and_empty:
            break
            
        if t > 5000:
            invalid_jobs_count += 1
            break
            
        # 1. Transitions
        for amr in AMR_KEYS:
            s = amr_states[amr]
            
            if s['mode'] == 'idle':
                if len(amr_queues[amr]) > 0:
                    s['job'] = amr_queues[amr][0]
                    mat = s['job'].type_
                    if inventory[amr][mat] == 0:
                        s['mode'] = 'moving_supply'
                        s['goal'] = SUPPLY_LOCATIONS[mat]
                        if need_log: 
                            queue_infos.append((s['job'].idx, t))
                            s['route_start'] = t
                    else:
                        s['mode'] = 'moving_station'
                        s['goal'] = STATIONS[s['job'].station]
                        if need_log and 'route_start' not in s: s['route_start'] = t
                else:
                    if positions[amr] != AMR_STARTS[amr]:
                        s['mode'] = 'moving_base'
                        s['goal'] = AMR_STARTS[amr]
                        if need_log: s['route_start'] = t
                        s['job'] = None
                        
            elif s['mode'] == 'processing':
                s['proc_ticks'] -= 1
                if s['proc_ticks'] <= 0:
                    # Done
                    mat = s['job'].type_
                    inventory[amr][mat] -= 1
                    station_occupied[s['job'].station] = False
                    
                    if need_log:
                        timelines.append((amr, t - s['job'].duration, t, f"process_{mat}", f"Job{s['job'].idx} {mat}({int(s['job'].duration)}s)"))
                        
                    amr_queues[amr].popleft()
                    s['mode'] = 'idle'
                    s['job'] = None
                    s['goal'] = None
                    if need_log: s['route_start'] = t

        # Need to immediately evaluate idle transitions again in case they just finished processing
        for amr in AMR_KEYS:
            s = amr_states[amr]
            if s['mode'] == 'idle':
                if len(amr_queues[amr]) > 0:
                    s['job'] = amr_queues[amr][0]
                    mat = s['job'].type_
                    if inventory[amr][mat] == 0:
                        s['mode'] = 'moving_supply'
                        s['goal'] = SUPPLY_LOCATIONS[mat]
                        if need_log: s['route_start'] = t
                    else:
                        s['mode'] = 'moving_station'
                        s['goal'] = STATIONS[s['job'].station]
                        if need_log and 'route_start' not in s: s['route_start'] = t
                else:
                    if positions[amr] != AMR_STARTS[amr]:
                        s['mode'] = 'moving_base'
                        s['goal'] = AMR_STARTS[amr]
                        if need_log: s['route_start'] = t
                        s['job'] = None

        # 2. Movement Negotiation
        moves = {}
        def prio(amr):
            m = amr_states[amr]['mode']
            if m == 'processing': return 100
            if m == 'idle': return 10
            if m == 'moving_station': return 50
            if m == 'moving_supply': return 40
            return 30
            
        ordered = sorted(AMR_KEYS, key=prio, reverse=True)
        reserved = set()
        
        # lock in Processing and Idles already at goal
        for amr in ordered:
            m = amr_states[amr]['mode']
            p = positions[amr]
            if m == 'processing':
                moves[amr] = p
                reserved.add(p)
            elif m == 'idle' and p == amr_states[amr].get('goal', p):
                moves[amr] = p
                reserved.add(p)
            elif m == 'moving_station' and p == amr_states[amr]['goal']:
                # Arrived but station occupied
                moves[amr] = p
                reserved.add(p)
                
        for amr in ordered:
            if amr in moves: continue
            
            p = positions[amr]
            g = amr_states[amr]['goal']
            
            # Simple A* next step
            path = shortest_path(p, g)
            next_step = path[1] if len(path) > 1 else p
            
            # Request move
            if next_step in reserved:
                moves[amr] = p
                reserved.add(p)
            else:
                swap = False
                for o_amr, o_next in moves.items():
                    if o_next == p and positions[o_amr] == next_step:
                        swap = True
                        break
                if swap:
                    moves[amr] = p
                    reserved.add(p)
                else:
                    moves[amr] = next_step
                    reserved.add(next_step)
                    
        # Apply moves and Evaluate arrivals
        for amr in AMR_KEYS:
            positions[amr] = moves[amr]
            if need_log and moves[amr] != path_logs[amr][-1]:
                path_logs[amr].append(moves[amr])
                
            s = amr_states[amr]
            p = positions[amr]
            
            if s['mode'] == 'moving_supply' and p == s['goal']:
                mat = s['job'].type_
                inventory[amr][mat] = 3
                if need_log:
                    dur = t - s.get('route_start', t)
                    timelines.append((amr, s.get('route_start', t), t, "supply", f"Replenish {mat}"))
                s['mode'] = 'idle'  # Re-evaluated next tick
                
            elif s['mode'] == 'moving_station' and p == s['goal']:
                # Are we the only one in the station? Yes, since it's a unique tile and we are on it.
                if not station_occupied[s['job'].station]:
                    station_occupied[s['job'].station] = True
                    s['mode'] = 'processing'
                    s['proc_ticks'] = s['job'].duration
                    if need_log:
                        dur = t - s.get('route_start', t)
                        if dur > 0: timelines.append((amr, s['route_start'], t, "travel", f"Job{s['job'].idx} trans {dur}s"))
                
            elif s['mode'] == 'moving_base' and p == s['goal']:
                s['mode'] = 'idle'
                if need_log:
                    dur = t - s.get('route_start', t)
                    if dur > 0: timelines.append((amr, s['route_start'], t, "return", f"Return {dur}s"))

        # End of tick
        t += 1
        
    avail = {a: float(t) for a in AMR_KEYS}
    return avail, timelines, queue_infos, path_logs, invalid_jobs_count

if __name__ == "__main__":
    from GA import make_jobs, greedy_individual, plot_gantt
    jobs = make_jobs()
    ind = greedy_individual(jobs)
    print("Testing GA_tick.py standalone...")
    ans = decode_schedule_tick_by_tick(ind, jobs, True, True)
    print(f"Time: {max(ans[0].values())}")
    # plot_gantt(ans[1], ans[2], jobs)
    print("Success")
