# _unused

Files that the current DFJSSP pipeline does not use, moved here instead of deleted.
Each file keeps its original path under this folder, so to restore one, move it back
to the same relative location (e.g. `_unused/Static_alogorithm/MILP` -> `Static_alogorithm/MILP`).

Active pipeline (kept in place): `run_all_demos.py` -> `Dynamic_/Periodic_Pairing_Demo.py`
-> `GNN_DDQN_V7.py` -> `Static_alogorithm/GA/GA.py` + `GNN/GNN.py` + `gnn_scheduler_best.pth`,
`AMRs_Routing/Routing_Demo.py`, `Testing_Tools/test_performance*.py`, `GNN_DDQN_V8.py`,
checkpoints `v7`, `v7_demo`, `v8_demo`, all `test_case/static` inboxes, `test_dataset_demo*.jsonl`.

## What's here and why
- **Old model versions**: `GNN_DDQN_GA_Optimizer`, `GA_V2`–`GA_V5`, `V6` (broken import), `GA_V7`, `scheduler*.py` — superseded by V7/V8; nothing imports them.
- **Old checkpoints**: `models_pth/gnn_ddqn_model_v2`–`v6` (~30 MB).
- **Static GA variants**: `GA_with_collisions.py` (+ its summary CSV), `GA_Routing_Output.py` (+ `amr_routing*.jsonl`, `amr_schedule*.jsonl` outputs).
- **Old GNN weights**: `gnn_scheduler_best_v1.pth`.
- **MILP/**: Gurobi/brute-force baselines for an older 3-station layout; inputs missing.
- **One-off tools**: `plot_intervals.py` (hard-coded log path), `analyze_job_types.py`, `fix.py` + `fix_period_comparison.*`.
- **Unused datasets**: `test_dataset_r2/r4`, `training_dataset_r1/r2/r4`.
- **Leftovers**: empty `schedule_outbox.jsonl` files, `.vscode/` (another machine's C/C++ config), `__pycache__/` (Python rebuilds these).

`MOVED.txt` is the log of exactly what was moved; `_move_unused.ps1` is the script that did it.
