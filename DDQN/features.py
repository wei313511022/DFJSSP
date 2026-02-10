from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from env import Coord, TaskSchedulingEnv


def flatten_jobs(scenario: Union[dict, List[dict]]) -> List[dict]:
    if isinstance(scenario, dict) and "jobs" in scenario:
        jobs = scenario.get("jobs", [])
        return jobs if isinstance(jobs, list) else []
    if isinstance(scenario, list):
        if len(scenario) == 0:
            return []
        if isinstance(scenario[0], dict) and "jobs" in scenario[0]:
            out: List[dict] = []
            for rec in scenario:
                jobs = rec.get("jobs", [])
                if isinstance(jobs, list):
                    out.extend(jobs)
            return out
        return scenario
    return []


def decode_robot_pos(state_vec: np.ndarray, num_robots: int, rid: int) -> Coord:
    """
    state layout:
      [onehot(3), n_tasks(1), t(1), robots(18), stations(5)]
    robots block starts at idx = 5
    each robot: [free_time, x, y, invA, invB, invC]
    """
    _ = num_robots
    base = 5 + rid * 6
    x = int(state_vec[base + 1])
    y = int(state_vec[base + 2])
    return (x, y)


def decode_robot_free_time(state_vec: np.ndarray, rid: int) -> float:
    base = 5 + rid * 6
    return float(state_vec[base + 0])


def decode_robot_inventory(state_vec: np.ndarray, num_robots: int, rid: int) -> Dict[str, int]:
    _ = num_robots
    base = 5 + rid * 6
    inv_a = int(state_vec[base + 3])
    inv_b = int(state_vec[base + 4])
    inv_c = int(state_vec[base + 5])
    return {"A": inv_a, "B": inv_b, "C": inv_c}


def decode_now_t(state_vec: np.ndarray) -> float:
    return float(state_vec[4])


def decode_station_busy(state_vec: np.ndarray) -> Dict[str, float]:
    return {
        "S1": float(state_vec[-5]),
        "S2": float(state_vec[-4]),
        "S3": float(state_vec[-3]),
        "S4": float(state_vec[-2]),
        "S5": float(state_vec[-1]),
    }


def build_actions_for_tasks(
    tasks: List[dict],
    inventory: Dict[str, int],
    capacity_per_type: int,
    allow_proactive_replenish: bool = True,
) -> List[Tuple[int, int]]:
    """
    Action is (task_idx, replenish_amount).
    - When inventory is 0, replenish must be >= 1.
    - When inventory > 0:
      - if allow_proactive_replenish=True: replenish can be 0..max_add
      - if False: replenish is forced to 0
    """
    actions: List[Tuple[int, int]] = []
    for idx, task in enumerate(tasks):
        jtype = str(task.get("type", "")).upper()
        inv = int(inventory.get(jtype, 0))
        max_add = max(0, capacity_per_type - inv)
        if inv > 0:
            min_add = 0
            if not allow_proactive_replenish:
                max_add = 0
        else:
            min_add = 1
        if max_add < min_add:
            continue
        for add in range(min_add, max_add + 1):
            actions.append((idx, add))
    return actions


def action_features_from_snapshot(
    env: TaskSchedulingEnv,
    robot_pos: Coord,
    now_t: float,
    station_busy: Dict[str, float],
    inventory: Dict[str, int],
    task: dict,
    replenish: int,
) -> np.ndarray:
    pickup = task["pickup"]
    drop = task["drop"]
    station = task["station"]
    proc = float(task["proc_time"])
    jtype = str(task.get("type", "")).upper()

    inv = inventory.get(jtype, 0)
    has_item = inv > 0
    if replenish > 0 or not has_item:
        travel = float(env._dist(robot_pos, pickup) + env._dist(pickup, drop))
    else:
        travel = float(env._dist(robot_pos, drop))
    arrive = now_t + travel
    wait = max(0.0, station_busy[station] - arrive)

    return np.array([travel, wait, proc, float(replenish)], dtype=np.float32)


def q_values_batch(
    policy_net: nn.Module, state: np.ndarray, feats: np.ndarray, device: torch.device
) -> torch.Tensor:
    """
    state: (state_dim,)
    feats: (K, action_dim)
    return: q (K,) on device
    """
    k = feats.shape[0]
    s_mat = np.repeat(state[None, :], k, axis=0).astype(np.float32)
    sa = np.concatenate([s_mat, feats], axis=1).astype(np.float32)
    sa_t = torch.from_numpy(sa).to(device)
    q = policy_net(sa_t).squeeze(1)
    return q
