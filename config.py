from __future__ import annotations
from typing import Dict, Set

TIME_LIMIT = 300

# Grid
GRID_SIZE: int = 10
BARRIER_NODES: Set[int] = {61, 62, 63, 65, 66, 67, 69, 70}

# Material
TYPE_TO_MATERIAL_NODE: Dict[str, int] = {"A": 8, "B": 5, "C": 2}
MATERIAL_PICK_QTY: int = 3
P_NODES = set(TYPE_TO_MATERIAL_NODE.values())

# Stations (station id -> delivery node)
JSON_STATION_MAPPING: Dict[int, int] = {5: 91, 4: 93, 3: 95, 2: 97, 1: 99}

# AMRs
M_SET = range(1, 4)
S_m: Dict[int, int] = {1: 8, 2: 5, 3: 2}

# Files
INBOX = "dispatch_inbox.jsonl"
SCHEDULE_OUTBOX = "Random_Job_Arrivals/schedule_outbox.jsonl"


def validate_fixed_nodes() -> None:
    fixed_nodes_to_check = set(S_m.values()) | set(TYPE_TO_MATERIAL_NODE.values()) | set(
        JSON_STATION_MAPPING.values()
    )
    bad_fixed = sorted(int(n) for n in fixed_nodes_to_check if int(n) in BARRIER_NODES)
    if bad_fixed:
        raise ValueError(
            f"Barrier nodes overlap with fixed start/pickup/delivery nodes: {bad_fixed}"
        )


validate_fixed_nodes()
