# DFJSSP — AMR scheduling with GA, GNN and GNN-DDQN

Research code for a **dynamic flexible job-shop scheduling problem with autonomous mobile robots (AMRs)**. Three robots on a grid fetch material, carry jobs to five workstations and process them there. The project answers two questions:

1. **Static:** given a batch of jobs, which robot does each job, and in what order? Solved with a genetic algorithm (GA) and a graph neural network (GNN).
2. **Dynamic:** jobs keep arriving over time. *When* should the schedule be recomputed? Solved with a GNN-DDQN agent that chooses WAIT or RESCHEDULE at every step and is compared with rescheduling on a fixed timer.

Old versions and one-off scripts are kept in `_unused/` (see `_unused/README.md`). Nothing outside that folder depends on them.

## The scenario

| Item | Value |
|---|---|
| Grid | x 0–9, y 1–9, 14 obstacle cells |
| Robots | AMR1 (2, 8), AMR2 (2, 5), AMR3 (2, 2); start with 3 units of A, B and C respectively |
| Stations | station1–5 at (9, 9), (9, 7), (9, 5), (9, 3), (9, 1); one robot at a time |
| Supply points | A (0, 8), B (0, 5), C (0, 2); a refill loads 3 units of that type |
| Job types | A = 5 s, B = 10 s, C = 25 s of processing |
| Makespan | Time until the last robot is back at its base |

All of these are defined at the top of `Static_alogorithm/GA/GA.py` and imported everywhere else.

## Folder map

```
DFJSSP/
├── run_all_demos.py            Launches the 4-window live demo
├── sync.txt                    Shared play/pause switch for the demo windows
├── AMRs_Routing/
│   └── Routing_Demo.py         Route-map viewer (display only)
├── Random_Job_Arrivals/
│   ├── models/                 Dynamic scheduler + demos
│   ├── models_pth/             Trained GNN-DDQN weights and benchmark results
│   └── Testing_Tools/          AI vs fixed-timer benchmark
├── Static_alogorithm/
│   ├── GA/                     Genetic algorithm + simulators (core library)
│   └── GNN/                    Static scheduling GNN, training, results
└── test_case/
    ├── static/                 Job batches for the static solvers
    └── dynamic/                Job-arrival episodes for the dynamic scheduler
```

## Requirements

Python 3.10+ with `torch`, `numpy` and `matplotlib`. A GPU is optional; everything runs on CPU. **Run each script from its own folder:** paths inside the scripts are relative.

## Quick start

| Goal | Folder | Command |
|---|---|---|
| Watch the live demo | `DFJSSP/` | `python run_all_demos.py`, then press SPACE in any window |
| Solve static batches with the GA | `Static_alogorithm/GA/` | `python GA.py --inbox ../../test_case/static/dispatch_inbox_30.jsonl --gantt` |
| Solve static batches with the GNN | `Static_alogorithm/GNN/` | `python GNN.py --inbox ../../test_case/static/dispatch_inbox_30.jsonl` |
| Train the static GNN | `Static_alogorithm/GNN/` | `python train.py --minutes 60` |
| Train the dynamic agent | `Random_Job_Arrivals/models/` | `python GNN_DDQN_V7.py` (settings in its `CONFIG` block) |
| Benchmark AI vs fixed timer | `Random_Job_Arrivals/Testing_Tools/` | `python test_performance.py --model ../models_pth/<dir>/<file>.pth` |

## Files

### Top level

| File | What it does |
|---|---|
| `run_all_demos.py` | Opens four windows side by side: the AI (dynamic) scheduler and the fixed-timer (periodic) scheduler, each with its own route map. Resets `sync.txt` and the `*_amr_state.json` files on start. |
| `sync.txt` | One character, `0` = paused, `1` = running. Every demo window reads it about every 200 ms; SPACE in any window flips it, so all four stay in step. |
| `.gitignore` | Ignores `venv/`. |

### `AMRs_Routing/`

| File | What it does |
|---|---|
| `Routing_Demo.py` | Draws robot positions, planned routes and status from a `*_amr_state.json` file written by a pairing demo. It does no simulation itself. Options: `--state_file`, `--window_pos`, `--title`, `--sync_file`. |

### `Random_Job_Arrivals/models/` — dynamic scheduling

| File | What it does |
|---|---|
| `GNN_DDQN_V7.py` | The dynamic environment and agent. Jobs arrive over time; at each step a GNN + dueling Double-DQN chooses WAIT or RESCHEDULE. On RESCHEDULE, the static GNN (`gnn_scheduler_best.pth`) builds a plan and GA local search improves it. Running the file trains the agent; `CONFIG` at the top sets dataset, save folder, episodes and GA settings. |
| `Dynamic_Pairing_Demo.py` | Live demo where the trained V7 agent decides when to reschedule. Loads `models_pth/gnn_ddqn_model_v7_demo/gnn_ddqn_model_v7_ep800.pth` and `test_case/dynamic/test_dataset_demo.jsonl`. Writes `dynamic_amr_state.json` (for the route map) and `dynamic_schedule_outbox.jsonl`. |
| `Periodic_Pairing_Demo.py` | Same demo, but reschedules every 10 s (`FIX_PERIOD`) instead of asking the agent. Writes `periodic_amr_state.json` and `periodic_schedule_outbox.jsonl`. |
| `dynamic_amr_state.json`, `periodic_amr_state.json` | Robot state snapshots written by the demos each tick and read by `Routing_Demo.py`. Regenerated on every run; a leftover `.tmp` copy is safe to delete. |

### `Random_Job_Arrivals/models_pth/` — dynamic agent weights

| Folder | Contents |
|---|---|
| `gnn_ddqn_model_v7/` | V7 checkpoints every 100 episodes (`_ep0` … `_ep900`, final `gnn_ddqn_model_v7.pth`) and benchmark CSVs/plots. |
| `gnn_ddqn_model_v7_demo/` | The V7 run used by the demo (`_ep800`), plus `_stable` and its benchmark results. |

Benchmark summaries recorded before the GA.py fixes: `benchmark_makespan_results_summary.csv` shows −12% makespan and −65% GA time against the fixed timer. Re-run before comparing with new results.

### `Random_Job_Arrivals/Testing_Tools/`

| File | What it does |
|---|---|
| `test_performance.py` | Runs every episode of a dataset twice, once with the trained agent and once with a fixed 10 s timer. Reports jobs finished, flow time, makespan and GA time, writes a per-episode CSV, a `_summary.csv` and a plot. Options: `--model`, `--output`, `--module` (default `models.GNN_DDQN_V7`). |

### `Static_alogorithm/GA/` — genetic algorithm (core library)

| File | What it does |
|---|---|
| `GA.py` | Scenario constants, A* and space-time A* routing, two schedule evaluators, and the GA. A solution is a job order plus a robot per job. Fast evaluator: `decode_schedule` (collision-free, per-station booking calendar). Real evaluator: `decode_schedule_tick_by_tick` (1-second steps, collisions, dodging, deadlock detection). `evolve()` runs 200 schedules × 150 generations, then local search. Run directly: `--inbox`, `--gantt`, `--save_img`; `DISPATCH_EVENT_INDEX=3` runs only batch 3. Writes `GA_summary_results.csv`. |
| `test_ga_performance.py` | Sweeps population size and number of generations on the first 10 batches; writes `ga_performance_summary.csv` and a plot. |
| `GA_summary_results.csv` | Output of the last `GA.py` run: makespan and computation time per batch. |
| `ga_performance_summary.csv` | Output of `test_ga_performance.py`. |
| `summary_results.csv` | Older GA results (60-job batches, before the GA.py fixes). |
| `collision_iters_compare/` | Earlier sweep of collision-aware local-search iterations: summary CSV and plot. |

### `Static_alogorithm/GNN/` — static scheduling GNN

| File | What it does |
|---|---|
| `GNN.py` | `SchedulerGNN`: robots and jobs as graph nodes with attention layers; at each step it scores every (robot, job) pair and picks one. `solve_with_gnn()` builds a full schedule this way. Run directly to solve batches: GNN schedule → 1,000 local-search steps → collision-aware step → real simulation. Loads `gnn_scheduler_best.pth` from the current folder. Options: `--inbox`, `--gantt`, `--save_img`, `--collision_iters`, `--output_csv`. |
| `train.py` | Trains `SchedulerGNN` with REINFORCE: 16 schedules per batch are sampled together and rewarded by their real-simulation makespan relative to their average. Trains on lines 0–899 of `dispatch_inbox_60`, keeps the checkpoint with the best score on lines 900–949; lines 950–999 stay unseen for testing. Options: `--minutes`, `--batch`, `--lr`, `--init` (continue from weights), `--out`, `--log`. |
| `test_gnn_collision_iters.py` | Runs `GNN.py` with collision iterations 0, 100, 500, 1000 and 2000 and plots makespan and time for each. |
| `gnn_scheduler_best.pth` | Original weights. **Loaded by `GNN.py` and by `GNN_DDQN_V7.py`**, so replacing this file changes both. Trained before the GA.py simulator fixes. |
| `gnn_scheduler_retrained.pth` | Weights from `train.py` (90 min on CPU, trained against the fixed simulator). To use them, rename to `gnn_scheduler_best.pth`. |
| `train_log.csv` | Learning curve of that training run: validation makespan every 25 steps. |
| `gnn_vs_ga_results.csv` | Per-batch comparison of the GA, the original GNN and the retrained GNN (see Results). |
| `GNN_summary_results.csv` | Output of the last `GNN.py` run. |

### `test_case/` — data and generators

| File | Contents |
|---|---|
| `static/dispatch_inbox_N.jsonl` | Job batches with N jobs each (N = 10, 20, 30, 40, 50, 60, 70, 80). Every file has 10 batches except `_60`, which has 1,000 (used for GNN training). |
| `static/Random_Job_Generator.py` | Writes new batch files: `python Random_Job_Generator.py -b 10 -s 30` → 10 batches of 30 jobs. |
| `dynamic/training_dataset_r1/r2/r4.jsonl` | Training episodes for the dynamic agent. The `rN` suffix is the mean time between job arrivals in seconds. |
| `dynamic/test_dataset_r2/r4.jsonl` | 100 test episodes each, about 100 jobs per episode. |
| `dynamic/test_dataset_demo.jsonl` | The single 145-job episode used by the demos and `test_performance.py`. |
| `dynamic/Random_Job_Arrivals.py` | Generates episodes with random (Poisson) arrivals; edit the constants at the top, then run. |

**Formats.** Static batch, one per line: `{"dispatch_time": …, "jobs": [{"jid", "type", "proc_time", "station"}]}`. Dynamic episode, one per line: `{"episode_id": …, "jobs": [{"id", "type", "arrival_time", "dest_station_id"}]}`.

## Notes

- **Results from before the GA.py fixes are not comparable.** Earlier versions of `GA.py` assigned jobs to the wrong robots in the real simulator; `summary_results.csv`, `GNN_summary_results.csv` and the V7 benchmark CSVs predate the fix.
- **`gnn_scheduler_best.pth` is shared.** The dynamic agent's environment loads it, so retraining V7 after swapping in new weights is recommended.
- **`test_performance.py` works with V7 models.** The more general benchmark variants are in `_unused/Random_Job_Arrivals/Testing_Tools/`.
