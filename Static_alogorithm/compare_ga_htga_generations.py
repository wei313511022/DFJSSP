"""Compare GA with the paper-based, AMR-adapted HTGA on dispatch events."""

import csv
import importlib.util
import multiprocessing as mp
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GA_PATH = Path(__file__).parent / "GA" / "GA.py"
HTGA_PATH = Path(__file__).parent / "Taguchi-GA" / "HTGA.py"
INBOX_DIR = ROOT / "test_case" / "static"
OUTPUT_PATH = Path(__file__).parent / "GA_vs_HTGA_generation20.csv"
INBOX_COUNTS = (10, 20, 30, 40)
BASE_SEED = 42
WORKER_MODULES = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def makespan(module, individual, jobs) -> float:
    availability, _, _, _, _ = module.decode_schedule_tick_by_tick(
        individual, jobs, need_log=False, check_collision=True
    )
    return float(max(availability.values()))


def init_worker() -> None:
    global WORKER_MODULES
    WORKER_MODULES = (
        load_module("benchmark_ga_worker", GA_PATH),
        load_module("benchmark_htga_worker", HTGA_PATH),
    )


def run_pair(task):
    inbox_jobs, event_index, job_data = task
    ga, htga = WORKER_MODULES
    ga_jobs = [ga.Job(idx=i, type_=kind, duration=duration, station=station)
               for i, (kind, duration, station) in enumerate(job_data)]
    htga_jobs = [htga.Job(idx=i, type_=kind, duration=duration, station=station)
                 for i, (kind, duration, station) in enumerate(job_data)]

    # Pair the algorithms on the same event and random seed.
    seed = BASE_SEED + int(event_index)
    random.seed(seed)
    ga_individual, _ = ga.evolve(ga_jobs)
    ga_makespan = makespan(ga, ga_individual, ga_jobs)

    random.seed(seed)
    htga_individual, _ = htga.evolve(htga_jobs)
    htga_makespan = makespan(htga, htga_individual, htga_jobs)
    delta = htga_makespan - ga_makespan
    improvement = ((ga_makespan - htga_makespan) / ga_makespan * 100.0
                   if ga_makespan else 0.0)
    return {
        "Inbox_Jobs": inbox_jobs,
        "Event_Index": event_index,
        "Job_Count": len(job_data),
        "Generations": 20,
        "GA_Makespan": f"{ga_makespan:.2f}",
        "HTGA_Makespan": f"{htga_makespan:.2f}",
        "HTGA_minus_GA": f"{delta:.2f}",
        "HTGA_Improvement_Percent": f"{improvement:.2f}",
    }


def main() -> None:
    ga = load_module("benchmark_ga", GA_PATH)
    htga = load_module("benchmark_htga", HTGA_PATH)
    if ga.GENERATIONS != 20 or htga.GENERATIONS != 20:
        raise RuntimeError("Both algorithms must use GENERATIONS = 20")

    columns = [
        "Inbox_Jobs",
        "Event_Index",
        "Job_Count",
        "Generations",
        "GA_Makespan",
        "HTGA_Makespan",
        "HTGA_minus_GA",
        "HTGA_Improvement_Percent",
    ]
    tasks = []
    for inbox_jobs in INBOX_COUNTS:
        inbox = INBOX_DIR / f"dispatch_inbox_{inbox_jobs}.jsonl"
        for event in ga.load_dispatch_events(inbox):
            job_data = [(job.type_, job.duration, job.station) for job in event["jobs"]]
            tasks.append((inbox_jobs, event["index"], job_data))

    completed = 0
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=columns)
        writer.writeheader()
        result_file.flush()
        worker_count = min(12, mp.cpu_count())
        with mp.Pool(worker_count, initializer=init_worker) as pool:
            for result in pool.imap_unordered(run_pair, tasks, chunksize=1):
                writer.writerow(result)
                result_file.flush()
                completed += 1
                print(f"[{completed}/{len(tasks)}] inbox={result['Inbox_Jobs']}, "
                      f"event={result['Event_Index']}: GA={result['GA_Makespan']}, "
                      f"HTGA={result['HTGA_Makespan']}", flush=True)

    print(f"Wrote {completed} comparisons to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
