#!/usr/bin/env python3
"""DPR-1 (type-based)

- Select job type by the largest remaining total processing time.
- Operations have no precedence (can be scheduled in any order).
- Assign AMR by earliest completion time (includes travel, station wait, and proc_time).
"""

from __future__ import annotations

import json
import os
import random
import time as pytime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from collections import deque
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch


# ===================== FIELD CONFIG =====================

TIME_LIMIT = None
GRID_SIZE = 10
TYPE_TO_MATERIAL_NODE = {"A": 8, "B": 5, "C": 2}
MATERIAL_PICK_QTY = 3  # each pickup replenishes 3 units of the same material
P_NODES = set(TYPE_TO_MATERIAL_NODE.values())

JSON_STATION_MAPPING = {5: 91, 4: 93, 3: 95, 2: 97, 1: 99}  # delivery nodes

M_SET = range(1, 4)  # 3 AMRs
S_m = {1: 2, 2: 5, 3: 8}  # AMR start nodes

# Barrier nodes (AMRs cannot traverse these grid nodes)
BARRIER_NODES = {61, 62, 63, 65, 66, 67, 69, 70}

# Basic validation: key nodes must not be barriers
_fixed_nodes_to_check = set(S_m.values()) | set(TYPE_TO_MATERIAL_NODE.values()) | set(
	JSON_STATION_MAPPING.values()
)
_bad_fixed = sorted(int(n) for n in _fixed_nodes_to_check if int(n) in BARRIER_NODES)
if _bad_fixed:
	raise ValueError(
		f"Barrier nodes overlap with fixed start/pickup/delivery nodes: {_bad_fixed}"
	)

# Paths
INBOX = "dispatch_inbox.jsonl"
SCHEDULE_OUTBOX = "Random_Job_Arrivals/schedule_outbox.jsonl"

def _neighbors(node: int, grid_size: int) -> Iterable[int]:
	r, c = (node - 1) // grid_size, (node - 1) % grid_size
	if r > 0:
		yield node - grid_size
	if r < grid_size - 1:
		yield node + grid_size
	if c > 0:
		yield node - 1
	if c < grid_size - 1:
		yield node + 1


def _calculate_distance_uncached(
	n1: int, n2: int, *, grid_size: int, barriers: Iterable[int]
) -> int:
	if n1 == n2:
		return 0

	barriers_set = set(int(b) for b in barriers)
	if n1 in barriers_set or n2 in barriers_set:
		raise ValueError(f"No path: start/end is a barrier (start={n1}, end={n2})")

	max_node = grid_size * grid_size
	if not (1 <= n1 <= max_node and 1 <= n2 <= max_node):
		raise ValueError(
			f"Node out of bounds for grid {grid_size}x{grid_size}: {n1}, {n2}"
		)

	q: deque[int] = deque([n1])
	dist: Dict[int, int] = {n1: 0}
	while q:
		cur = q.popleft()
		d = dist[cur]
		for nb in _neighbors(cur, grid_size):
			if nb in barriers_set or nb in dist:
				continue
			dist[nb] = d + 1
			if nb == n2:
				return dist[nb]
			q.append(nb)

	raise ValueError(
		f"No path found from {n1} to {n2} with barriers={sorted(barriers_set)}"
	)


@lru_cache(maxsize=None)
def _calculate_distance_cached(n1: int, n2: int) -> int:
	return _calculate_distance_uncached(n1, n2, grid_size=GRID_SIZE, barriers=BARRIER_NODES)


def calculate_distance(
	node1: int, node2: int, grid_size: int = GRID_SIZE, barriers: Optional[Iterable[int]] = None
) -> int:
	"""Shortest 4-neighbor grid distance avoiding barrier nodes."""
	n1 = int(node1)
	n2 = int(node2)
	if barriers is None or set(int(b) for b in barriers) == set(BARRIER_NODES):
		return _calculate_distance_cached(n1, n2)
	return _calculate_distance_uncached(n1, n2, grid_size=grid_size, barriers=barriers)


# ===================== DPR(a1) DISPATCHING LOGIC =====================


@dataclass
class Operation:
	jtype: str
	proc_time: float
	station: int
	delivery_node: int
	eligible_amrs: Optional[List[int]] = None  # if None, all AMRs eligible


@dataclass
class TaskState:
	jid: int
	release_time: float
	op: Operation


@dataclass
class AMRState:
	amr_id: int
	available_time: float
	current_node: int
	inv: Dict[str, int]


def _build_job_states_from_event(jobs_raw: List[dict], dispatch_time: float) -> List[TaskState]:
	# Backward-compatible name: build task list (each operation is an independent task).
	tasks: List[TaskState] = []
	for j in jobs_raw:
		jid = int(j.get("jid"))
		jtype_default = str(j.get("type", "A")).upper()
		proc_time_default = float(j.get("proc_time", 0.0))

		# NOTE: We ignore dispatch_time (all tasks are available immediately).
		release_time = 0.0

		if isinstance(j.get("operations"), list):
			for op in j["operations"]:
				st = int(op.get("station"))
				g_l = JSON_STATION_MAPPING.get(st)
				if g_l is None:
					raise ValueError(f"Unknown station {st} for jid={jid}")
				tasks.append(
					TaskState(
						jid=jid,
						release_time=float(release_time),
						op=Operation(
							jtype=str(op.get("type", jtype_default)).upper(),
							proc_time=float(op.get("proc_time", proc_time_default)),
							station=st,
							delivery_node=int(g_l),
							eligible_amrs=op.get("eligible_amrs"),
						),
					)
				)
		else:
			st = int(j.get("station"))
			g_l = JSON_STATION_MAPPING.get(st)
			if g_l is None:
				raise ValueError(f"Unknown station {st} for jid={jid}")
			tasks.append(
				TaskState(
					jid=jid,
					release_time=float(release_time),
					op=Operation(
						jtype=jtype_default,
						proc_time=proc_time_default,
						station=st,
						delivery_node=int(g_l),
					),
				)
			)
	return tasks


def _eligible_amrs_for_op(op: Operation) -> List[int]:
	if op.eligible_amrs is None:
		return [int(m) for m in M_SET]
	return [int(m) for m in op.eligible_amrs]


def _amr_is_material_compatible(amr: AMRState, op: Operation) -> bool:
	# With multi-type inventory, any AMR can serve any type; refill happens only when
	# that type's onboard qty is 0.
	return True


def _choose_next_type_by_remaining_time(
	tasks: List[TaskState],
	amrs: Dict[int, AMRState],
) -> Optional[str]:
	"""Pick job type with the largest remaining processing time among feasible types."""
	rem: Dict[str, float] = {}
	feasible_types = set()
	for tsk in tasks:
		t = str(tsk.op.jtype).upper()
		rem[t] = float(rem.get(t, 0.0)) + float(tsk.op.proc_time)
		eligible = _eligible_amrs_for_op(tsk.op)
		if any(_amr_is_material_compatible(amrs[m], tsk.op) for m in eligible):
			feasible_types.add(t)

	if not feasible_types:
		return None

	best = max(rem[t] for t in feasible_types)
	tied = [t for t in feasible_types if abs(rem[t] - best) <= 1e-12]
	return str(random.choice(tied)).upper()


def _estimate_required_station_wait(
	tsk: TaskState,
	amr: AMRState,
	station_available_time: Dict[int, float],
) -> float:
	"""Estimate this task's station-queue wait time if executed next on this AMR.

	Matches the wait_time definition used in _schedule_one_operation:
	wait_time = station_start - arrive_delivery (clipped at 0).
	"""
	op = tsk.op
	t = str(op.jtype).upper()

	prev_end_time = float(amr.available_time)
	prev_node = int(amr.current_node)
	op_ready_time = float(tsk.release_time)

	onboard_qty = int(amr.inv.get(t, 0))
	needs_refill = onboard_qty <= 0
	if needs_refill:
		pickup_node = int(TYPE_TO_MATERIAL_NODE[t])
		to_pick_travel = float(calculate_distance(prev_node, pickup_node))
		pick_time = prev_end_time + to_pick_travel
	else:
		pickup_node = prev_node
		pick_time = prev_end_time

	delivery_node = int(op.delivery_node)
	transport_time = float(calculate_distance(int(pickup_node), int(delivery_node)))
	arrive_delivery = pick_time + transport_time

	station_prev_end = float(station_available_time.get(delivery_node, 0.0))
	station_start = max(arrive_delivery, station_prev_end, op_ready_time)
	return float(max(0.0, station_start - arrive_delivery))


def _choose_task_of_type(
	tasks: List[TaskState],
	chosen_type: str,
	amrs: Dict[int, AMRState],
	station_available_time: Dict[int, float],
) -> Optional[TaskState]:
	"""Pick a task of the chosen type with deterministic tie-breaks.

	Rule:
	1) longest processing time (desc)
	2) smallest required waiting time (asc)
	3) smallest job id (asc)
	"""
	pool = [t for t in tasks if str(t.op.jtype).upper() == str(chosen_type).upper()]
	if not pool:
		return None

	best_tsk: Optional[TaskState] = None
	best_key: Optional[Tuple[float, float, int]] = None

	for tsk in pool:
		eligible = _eligible_amrs_for_op(tsk.op)
		eligible = [m for m in eligible if _amr_is_material_compatible(amrs[m], tsk.op)]
		if not eligible:
			continue

		# Compute the best (minimum) station-wait achievable among eligible AMRs.
		best_wait = float("inf")
		for m in eligible:
			try:
				w = _estimate_required_station_wait(tsk, amrs[m], station_available_time)
			except Exception:
				continue
			if w < best_wait:
				best_wait = w

		if best_wait == float("inf"):
			continue

		# Key to MINIMIZE; proc_time is negated to prefer larger.
		key = (-float(tsk.op.proc_time), float(best_wait), int(tsk.jid))
		if best_key is None or key < best_key:
			best_key = key
			best_tsk = tsk

	return best_tsk


def _estimate_completion_time(
	tsk: TaskState,
	amr: AMRState,
	station_available_time: Dict[int, float],
) -> Tuple[float, float, float]:
	"""Return (end_time, needs_refill_flag, dist_to_pickup) for this AMR on this task."""
	op = tsk.op
	t = str(op.jtype).upper()

	prev_end_time = float(amr.available_time)
	prev_node = int(amr.current_node)
	op_ready_time = float(tsk.release_time)

	onboard_qty = int(amr.inv.get(t, 0))
	needs_refill = onboard_qty <= 0
	if needs_refill:
		pickup_node = int(TYPE_TO_MATERIAL_NODE[t])
		pick_dist = float(calculate_distance(prev_node, pickup_node))
		to_pick_travel = float(pick_dist)
		arrive_pick = prev_end_time + to_pick_travel
		pick_time = arrive_pick
	else:
		pickup_node = prev_node
		to_pick_travel = 0.0
		pick_dist = 0.0
		pick_time = prev_end_time

	delivery_node = int(op.delivery_node)
	transport_time = float(calculate_distance(int(pickup_node), int(delivery_node)))
	arrive_delivery = pick_time + transport_time

	station_prev_end = float(station_available_time.get(delivery_node, 0.0))
	station_start = max(arrive_delivery, station_prev_end, op_ready_time)
	end_time = station_start + float(op.proc_time)

	return (float(end_time), float(needs_refill), float(pick_dist))


def _choose_amr_for_op_a1(
	js: TaskState,
	op: Operation,
	amrs: Dict[int, AMRState],
	station_available_time: Dict[int, float],
) -> int:
	eligible = _eligible_amrs_for_op(op)
	compatible = [m for m in eligible if _amr_is_material_compatible(amrs[m], op)]
	if not compatible:
		raise RuntimeError(f"No eligible AMR for jid={js.jid}, type={op.jtype}.")

	vals: List[Tuple[int, float, float, float]] = []
	
	for m in compatible:
		end_time, needs_refill, dist_to_pickup = _estimate_completion_time(js, amrs[int(m)], station_available_time)
		vals.append((int(m), end_time, needs_refill, dist_to_pickup))

	# Sort by (end_time, needs_refill, distance_to_pickup, amr_id)
	# Prefer: earlier completion, no refill needed, closer to pickup if refill needed, smaller AMR id
	vals.sort(key=lambda x: (x[1], x[2], x[3], x[0]))
	return int(vals[0][0])


def _schedule_one_operation(
	js: TaskState,
	op: Operation,
	amr: AMRState,
	station_available_time: Dict[int, float],
) -> dict:
	"""Schedule js.next op on a specific AMR and update states.

	Returns a record compatible with MILP_0108 plotting + outbox.
	"""

	t = str(op.jtype).upper()
	if t not in TYPE_TO_MATERIAL_NODE:
		raise ValueError(f"Unknown job type: {t}")

	prev_end_time = float(amr.available_time)
	prev_node = int(amr.current_node)

	# No precedence among operations inside a job.
	op_ready_time = float(js.release_time)

	# Inventory is tracked per type; each type max 3; refill only when that type is 0.
	onboard_qty = int(amr.inv.get(t, 0))
	needs_refill = onboard_qty <= 0
	if needs_refill:
		pickup_node = int(TYPE_TO_MATERIAL_NODE[t])
		to_pick_travel = float(calculate_distance(prev_node, pickup_node))
		arrive_pick = prev_end_time + to_pick_travel
		# Pick immediately upon arrival.
		pick_time = arrive_pick
	else:
		pickup_node = prev_node
		to_pick_travel = 0.0
		# If no refill, treat "pick" as the departure moment from the previous node.
		pick_time = prev_end_time

	idle_before_pick = max(0.0, pick_time - (prev_end_time + to_pick_travel))

	delivery_node = int(op.delivery_node)
	transport_time = float(calculate_distance(int(pickup_node), int(delivery_node)))
	arrive_delivery = pick_time + transport_time

	station_prev_end = float(station_available_time.get(delivery_node, 0.0))
	# Station processing start must respect:
	# - AMR arriving at the station
	# - station availability (no overlap)
	# - job precedence / op-ready time
	station_start = max(arrive_delivery, station_prev_end, op_ready_time)
	wait_time = max(0.0, station_start - arrive_delivery)

	end_time = station_start + float(op.proc_time)

	# Update station and AMR states
	station_available_time[delivery_node] = end_time
	amr.available_time = end_time
	amr.current_node = delivery_node

	# Update inventory
	if needs_refill:
		# Refill to the per-type cap
		amr.inv[t] = MATERIAL_PICK_QTY
	# Consume one unit of this type
	amr.inv[t] = max(0, int(amr.inv.get(t, 0)) - 1)

	return {
		"jid": int(js.jid),
		"type": t,
		"proc_time": float(op.proc_time),
		"assigned_agv": int(amr.amr_id),
		"pickup_node": int(pickup_node),
		"delivery_node": int(delivery_node),
		"pick_time": float(pick_time),
		"del_time": float(station_start),
		"end_time": float(end_time),
		"transport_time": float(transport_time),
		"wait_time": float(wait_time),
		# extra context for time-aligned rendering
		"prev_end_time": float(prev_end_time),
		"prev_node": int(prev_node),
		"to_pick_travel": float(to_pick_travel),
		"idle_before_pick": float(idle_before_pick),
		# metadata
		"policy": "type_remaining_time",
		"needs_refill": bool(needs_refill),
	}


def dispatch_a1_event(
	jobs_raw: List[dict],
	dispatch_time: float,
	amr_available_time: Dict[int, float],
	amr_current_node: Dict[int, int],
	station_available_time: Dict[int, float],
	amr_inventory: Dict[int, Dict[str, int]],
	*,
	seed: Optional[int] = None,
) -> List[dict]:
	"""Run dispatch for one batch; returns scheduled operation records.

	NOTE: AMR inventory persists across events (each task consumes 1 unit of its type;
	replenish that type to 3 only when its onboard qty reaches 0).
	"""
	if seed is not None:
		random.seed(int(seed))

	tasks = _build_job_states_from_event(jobs_raw, dispatch_time)

	amrs: Dict[int, AMRState] = {}
	for m in M_SET:
		amrs[int(m)] = AMRState(
			amr_id=int(m),
			available_time=float(amr_available_time.get(int(m), 0.0)),
			current_node=int(amr_current_node.get(int(m), S_m[int(m)])),
			# Inventory persists across events; copy to avoid aliasing globals during scheduling.
			inv={str(k).upper(): int(v) for k, v in dict(amr_inventory.get(int(m), {})).items()},
		)

	records: List[dict] = []
	while tasks:
		chosen_type = _choose_next_type_by_remaining_time(tasks, amrs)
		if chosen_type is None:
			break
		tsk = _choose_task_of_type(tasks, chosen_type, amrs, station_available_time)
		if tsk is None:
			# Should not happen with the current material rules.
			break
		op = tsk.op
		m = _choose_amr_for_op_a1(tsk, op, amrs, station_available_time)
		rec = _schedule_one_operation(tsk, op, amrs[m], station_available_time)
		records.append(rec)
		tasks.remove(tsk)

	if tasks:
		raise RuntimeError(
			"Dispatch could not schedule all tasks under the current constraints. "
			"This usually indicates an infeasible type/AMR/inventory configuration."
		)

	# Push AMR states back to globals (across events)
	for m in M_SET:
		amr_available_time[int(m)] = float(amrs[int(m)].available_time)
		amr_current_node[int(m)] = int(amrs[int(m)].current_node)
		# Persist inventory across events
		amr_inventory[int(m)] = dict(amrs[int(m)].inv)

	return records


# ===================== VISUAL CONFIG / STATE (match MILP_0108) =====================

AMR_COUNT = 3
UPDATE_INTERVAL_MS = 250

DISPATCHING_LEFT_LABEL_PAD = 7.25
QUEUE_LEFT_LABEL_PAD = 5.25
VIEW_WIDTH = 60.0

AX_Y_MIN, AX_Y_MAX = 0.0, 2.0
TOP_Y_CENTER = 1.25
TOP_LANE_H = 0.5

BOTTOM_MIN = 0.0
BOTTOM_HEIGHT = (AX_Y_MAX - AX_Y_MIN) / 2.0
AMR_Y_CENTERS = [
	BOTTOM_MIN + (i + 0.5) * (BOTTOM_HEIGHT / AMR_COUNT) for i in range(AMR_COUNT)
]
AMR_LANE_H = BOTTOM_HEIGHT / AMR_COUNT * 0.7

_cycle = plt.rcParams.get("axes.prop_cycle", None)
_cycle_list = _cycle.by_key()["color"] if _cycle else ["C0", "C1", "C2", "C3"]
TYPE_COLORS = {"A": _cycle_list[0], "B": _cycle_list[1], "C": _cycle_list[2]}
TRANSPORT_COLOR = "lightgray"
WAIT_COLOR = TRANSPORT_COLOR


@dataclass
class JobVisual:
	jid: int
	jtype: str
	proc_time: float


waiting: List[JobVisual] = []
rects_top: List[Rectangle] = []
texts_top: List = []

amr_rects: Dict[int, List[Rectangle]] = {i: [] for i in range(1, AMR_COUNT + 1)}
amr_texts: Dict[int, List] = {i: [] for i in range(1, AMR_COUNT + 1)}
amr_load: Dict[int, float] = {i: 0.0 for i in range(1, AMR_COUNT + 1)}

_lines_consumed = 0
current_makespan = 0.0
total_solve_time = 0.0

# Global resource state across events
agv_available_time: Dict[int, float] = {m: 0.0 for m in M_SET}
agv_current_node: Dict[int, int] = {m: int(S_m[m]) for m in M_SET}
station_available_time: Dict[int, float] = {int(n): 0.0 for n in JSON_STATION_MAPPING.values()}

# Material inventory across events per AMR (multi-type, capped per type)
amr_inventory: Dict[int, Dict[str, int]] = {m: {} for m in M_SET}


def remove_artists(rects, texts):
	for a in rects:
		try:
			a.remove()
		except Exception:
			pass
	for t in texts:
		try:
			t.remove()
		except Exception:
			pass
	rects.clear()
	texts.clear()


def rebuild_top_lane(ax):
	remove_artists(rects_top, texts_top)
	x = QUEUE_LEFT_LABEL_PAD
	for j in waiting:
		r = Rectangle(
			(x + 2.0, TOP_Y_CENTER),
			j.proc_time / 2.0,
			TOP_LANE_H,
			linewidth=1.2,
			edgecolor="black",
			facecolor=TYPE_COLORS.get(j.jtype, "C3"),
			clip_on=True,
		)
		ax.add_patch(r)
		rects_top.append(r)
		t = ax.text(
			x + j.proc_time / 4.0 + 2.0,
			TOP_Y_CENTER + TOP_LANE_H / 2.0,
			f"J_{j.jid}",
			ha="center",
			va="center",
			fontsize=9,
			weight="bold",
			clip_on=True,
		)
		texts_top.append(t)
		x += j.proc_time / 2.0


def draw_on_amr(
	ax,
	amr_id: int,
	j: JobVisual,
	transport_time: float = 0.0,
	pick_time: float = 0.0,
	del_time: float = 0.0,
	end_time: float = 0.0,
	pickup_node: int | None = None,
	delivery_node: int | None = None,
	wait_time: float = 0.0,
):
	y = AMR_Y_CENTERS[amr_id - 1]
	base_x = DISPATCHING_LEFT_LABEL_PAD
	scale = 1.0 / 2.5

	prev_end_time = float(getattr(j, "prev_end_time", 0.0))
	prev_node = getattr(j, "prev_node", None)
	to_pick_travel = float(getattr(j, "to_pick_travel", 0.0))
	idle_before_pick = float(getattr(j, "idle_before_pick", 0.0))

	pick_time = float(pick_time)
	del_time = float(del_time)
	end_time = float(end_time)
	transport_time = float(transport_time)
	wait_time = float(wait_time)

	t0 = prev_end_time
	t1 = t0 + max(0.0, to_pick_travel)
	t2 = pick_time
	t3 = t2 + max(0.0, transport_time)
	t4 = del_time
	t5 = end_time

	t1 = min(t1, t2)
	t3 = min(t3, t4)

	def _rect(t_start: float, t_end: float, *, face, hatch, alpha, z, label=None):
		w = max(0.0, (t_end - t_start)) * scale
		if w <= 0:
			return
		x0 = base_x + t_start * scale
		rseg = Rectangle(
			(x0, y - AMR_LANE_H / 2.0),
			w,
			AMR_LANE_H,
			linewidth=0.0,
			edgecolor="gray",
			facecolor=face,
			hatch=hatch,
			clip_on=True,
		)
		rseg.set_alpha(1.0)
		rseg.set_zorder(z)
		ax.add_patch(rseg)
		amr_rects[amr_id].append(rseg)
		if label and w > 1.0:
			txt = ax.text(
				x0 + w / 2.0,
				y,
				label,
				ha="center",
				va="center",
				fontsize=6,
				color="black",
				weight="bold",
				clip_on=True,
			)
			txt.set_zorder(4)
			amr_texts[amr_id].append(txt)

	_rect(
		t0,
		t1,
		face=TRANSPORT_COLOR,
		hatch="///",
		z=1,
		alpha=1.0,
		label=(
			(
				f"{int(prev_node)}→{int(pickup_node)}\n({(t1 - t0):.0f})"
				if (prev_node is not None and pickup_node is not None)
				else f"({(t1 - t0):.0f})"
			)
			if (t1 - t0) > 0
			else None
		),
	)
	_rect(
		t1,
		t2,
		face=WAIT_COLOR,
		hatch="..",
		z=1,
		alpha=1.0,
		label=f"W\n({(t2 - t1):.0f})" if (t2 - t1) > 0 else None,
	)

	_rect(
		t2,
		t3,
		face=TRANSPORT_COLOR,
		hatch="///",
		z=2,
		alpha=1.0,
		label=(
			(
				f"{int(pickup_node)}→{int(delivery_node)}\n({(t3 - t2):.0f})"
				if (pickup_node is not None and delivery_node is not None)
				else f"({(t3 - t2):.0f})"
			)
			if (t3 - t2) > 0
			else None
		),
	)
	_rect(
		t3,
		t4,
		face=WAIT_COLOR,
		hatch="..",
		alpha=1.0,
		z=2,
		label=f"W\n({(t4 - t3):.0f})" if (t4 - t3) > 0 else None,
	)

	proc_w = max(0.0, (t5 - t4)) * scale
	x_proc = base_x + t4 * scale
	r = Rectangle(
		(x_proc, y - AMR_LANE_H / 2.0),
		proc_w,
		AMR_LANE_H,
		linewidth=1.8,
		edgecolor="black",
		facecolor=TYPE_COLORS.get(j.jtype, "C3"),
		clip_on=True,
	)
	r.set_zorder(5)
	ax.add_patch(r)
	amr_rects[amr_id].append(r)

	node_label = f"D:{delivery_node}" if delivery_node else ""
	label_text = (
		f"J_{j.jid}\n({float(j.proc_time):.0f})\n{node_label}"
		if node_label
		else f"J_{j.jid}\n({float(j.proc_time):.0f})"
	)
	t = ax.text(
		x_proc + proc_w / 2.0,
		y,
		label_text,
		ha="center",
		va="center",
		fontsize=6,
		weight="bold",
		color="white",
		clip_on=True,
	)
	t.set_zorder(6)
	amr_texts[amr_id].append(t)

	amr_load[amr_id] += (
		float(j.proc_time)
		+ transport_time
		+ wait_time
		+ max(0.0, to_pick_travel)
		+ max(0.0, idle_before_pick)
	)


def draw_static_panels(ax):
	band_frac = 0.12
	top_panel = Rectangle(
		(0.0, 0.5),
		band_frac,
		0.5,
		transform=ax.transAxes,
		fill=False,
		linewidth=1.8,
		clip_on=False,
		zorder=3,
	)
	ax.add_patch(top_panel)
	tp = ax.text(
		band_frac * 0.5,
		0.75,
		"Dispatching\nQueue",
		transform=ax.transAxes,
		ha="center",
		va="center",
		fontsize=9,
		weight="bold",
		color="gray",
		clip_on=True,
		zorder=4,
	)
	tp.set_clip_path(top_panel)

	bot_panel = Rectangle(
		(0.0, 0.0),
		band_frac,
		0.5,
		transform=ax.transAxes,
		fill=False,
		linewidth=1.8,
		clip_on=False,
		zorder=3,
	)
	ax.add_patch(bot_panel)

	for i in range(AMR_COUNT):
		y_frac = (i + 0.5) / AMR_COUNT * 0.5
		txt = ax.text(
			band_frac * 0.5,
			y_frac,
			f"AMR {i+1}",
			transform=ax.transAxes,
			ha="center",
			va="center",
			fontsize=10,
			weight="bold",
			color="gray",
			clip_on=True,
			zorder=4,
		)
		txt.set_clip_path(bot_panel)

	handles = [
		Patch(facecolor=TYPE_COLORS[k], edgecolor="black", label=f"Type {k}")
		for k in TYPE_COLORS
	]
	handles.append(
		Patch(facecolor=TRANSPORT_COLOR, edgecolor="gray", hatch="///", label="Transportation")
	)
	handles.append(Patch(facecolor=WAIT_COLOR, edgecolor="gray", hatch="..", label="Waiting"))
	ax.legend(handles=handles, loc="upper right", frameon=True)


def update_title(ax):
	ax.set_title(
		f"AMR Scheduler with DPR-1 | Makespan: {current_makespan:.1f} | Solve: {total_solve_time:.2f}s"
	)


def process_event_line_visual(line: str, ax, out_f):
	global waiting, current_makespan, total_solve_time

	try:
		data = json.loads(line)
	except Exception:
		return

	dispatch_time = float(data.get("dispatch_time", 0.0))
	jobs_raw = data.get("jobs", [])
	if not jobs_raw:
		return

	waiting = []
	for j in jobs_raw:
		waiting.append(
			JobVisual(
				jid=int(j.get("jid")),
				jtype=str(j.get("type", "A")).upper(),
				proc_time=float(j.get("proc_time", 0.0)),
			)
		)
	rebuild_top_lane(ax)

	# Dispatch using DPR a1
	_t_wall_start = pytime.perf_counter()
	records = dispatch_a1_event(
		jobs_raw,
		dispatch_time=dispatch_time,
		amr_available_time=agv_available_time,
		amr_current_node=agv_current_node,
		station_available_time=station_available_time,
		amr_inventory=amr_inventory,
	)
	_t_wall = pytime.perf_counter() - _t_wall_start
	total_solve_time += float(_t_wall)
	if not records:
		return

	# Makespan over global timeline
	current_makespan = max(
		float(current_makespan),
		max(float(r.get("end_time", 0.0)) for r in records),
	)

	# Sort by time only (do not bias by jid); randomize exact-ties for visualization/output.
	records.sort(key=lambda r: float(r.get("pick_time", 0.0)))
	# random tie-break among identical pick_time
	_eps = 1e-12
	_i = 0
	while _i < len(records):
		_j = _i + 1
		t0 = float(records[_i].get("pick_time", 0.0))
		while _j < len(records) and abs(float(records[_j].get("pick_time", 0.0)) - t0) <= _eps:
			_j += 1
		if _j - _i > 1:
			sub = records[_i:_j]
			random.shuffle(sub)
			records[_i:_j] = sub
		_i = _j

	for rec in records:
		amr = int(rec.get("assigned_agv"))
		jid = int(rec.get("jid"))
		jtype = str(rec.get("type") or "?")
		proc_time = float(rec.get("proc_time", 0.0))
		transport_time = float(rec.get("transport_time", 0.0))
		wait_time = float(rec.get("wait_time", 0.0))
		pick_time = float(rec.get("pick_time", 0.0))
		del_time = float(rec.get("del_time", 0.0))
		end_time = float(rec.get("end_time", 0.0))
		pickup_node = rec.get("pickup_node")
		delivery_node = rec.get("delivery_node")

		station = None
		for st, del_node in JSON_STATION_MAPPING.items():
			if int(del_node) == int(delivery_node):
				station = st
				break

		vjob = JobVisual(jid=jid, jtype=jtype, proc_time=proc_time)
		try:
			vjob.prev_end_time = float(rec.get("prev_end_time", 0.0))
			vjob.prev_node = rec.get("prev_node")
			vjob.to_pick_travel = float(rec.get("to_pick_travel", 0.0))
			vjob.idle_before_pick = float(rec.get("idle_before_pick", 0.0))
		except Exception:
			pass

		draw_on_amr(
			ax,
			amr,
			vjob,
			transport_time=transport_time,
			pick_time=pick_time,
			del_time=del_time,
			end_time=end_time,
			pickup_node=pickup_node,
			delivery_node=delivery_node,
			wait_time=wait_time,
		)

		rec_out = {
			"generated_at": dispatch_time,
			"policy": "a1",
			"amr": amr,
			"jid": jid,
			"type": jtype,
			"proc_time": proc_time,
			"transport_time": transport_time,
			"station": str(station) if station is not None else "?",
			"pickup_node": pickup_node,
			"delivery_node": delivery_node,
		}
		out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
		out_f.flush()


def main():
	global _lines_consumed

	os.makedirs(os.path.dirname(SCHEDULE_OUTBOX), exist_ok=True)
	open(SCHEDULE_OUTBOX, "w", encoding="utf-8").close()

	fig, ax = plt.subplots(figsize=(16, 5.2))
	ax.set_ylim(AX_Y_MIN, AX_Y_MAX)
	ax.set_xlim(0.0, VIEW_WIDTH)
	ax.set_yticks([])
	ax.set_xticks([])
	for side in ("top", "right", "bottom", "left"):
		ax.spines[side].set_visible(True)

	draw_static_panels(ax)
	update_title(ax)

	def on_scroll(event):
		if event.inaxes != ax:
			return
		xlim = ax.get_xlim()
		xrange = xlim[1] - xlim[0]
		if event.button == "up":
			shift = -xrange * 0.1
		elif event.button == "down":
			shift = xrange * 0.1
		else:
			return
		ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
		fig.canvas.draw_idle()

	def on_key(event):
		xlim = ax.get_xlim()
		xrange = xlim[1] - xlim[0]

		if event.key == "left":
			shift = -xrange * 0.15
		elif event.key == "right":
			shift = xrange * 0.15
		elif event.key == "home":
			ax.set_xlim([0, xrange])
			fig.canvas.draw_idle()
			return
		else:
			return
		ax.set_xlim([xlim[0] + shift, xlim[1] + shift])
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("scroll_event", on_scroll)
	fig.canvas.mpl_connect("key_press_event", on_key)

	pan_data = {"pressed": False, "xpress": None, "xlim": None}

	def on_press(event):
		if event.inaxes != ax:
			return
		pan_data["pressed"] = True
		pan_data["xpress"] = event.xdata
		pan_data["xlim"] = list(ax.get_xlim())

	def on_release(event):
		pan_data["pressed"] = False
		pan_data["xpress"] = None

	def on_motion(event):
		if not pan_data["pressed"] or event.inaxes != ax or pan_data["xpress"] is None:
			return
		dx = event.xdata - pan_data["xpress"]
		new_xlim = [pan_data["xlim"][0] - dx, pan_data["xlim"][1] - dx]
		ax.set_xlim(new_xlim)
		fig.canvas.draw_idle()

	fig.canvas.mpl_connect("button_press_event", on_press)
	fig.canvas.mpl_connect("button_release_event", on_release)
	fig.canvas.mpl_connect("motion_notify_event", on_motion)

	timer = fig.canvas.new_timer(interval=UPDATE_INTERVAL_MS)

	def tick():
		global _lines_consumed
		changed = False

		try:
			with open(INBOX, "r", encoding="utf-8") as f_in, open(
				SCHEDULE_OUTBOX, "a", encoding="utf-8"
			) as f_out:
				lines = f_in.readlines()
				if _lines_consumed < len(lines):
					new_lines = lines[_lines_consumed :]
					_lines_consumed = len(lines)
					for ln in new_lines:
						if ln.strip():
							process_event_line_visual(ln, ax, f_out)
							changed = True
							update_title(ax)
							fig.canvas.draw_idle()
							fig.canvas.flush_events()
		except FileNotFoundError:
			pass

		if changed:
			update_title(ax)
			fig.canvas.draw_idle()

		timer.start()

	timer.add_callback(tick)
	timer.start()
	plt.show()


if __name__ == "__main__":
	main()

