"""
Train the static scheduling GNN (SchedulerGNN) against the GA.py simulator.

Method (the network and schedule-building rule are GNN.py's own build_schedules):
  * REINFORCE with a shared baseline: each step samples B schedules for ONE instance
    in a single batched forward pass; advantage = (mean makespan - own makespan) / std.
  * Reward = makespan of the RAW GNN schedule in the real tick-by-tick simulator
    (no local search inside the reward, so the signal reflects the GNN's own choices).
  * Train / validation / test split of dispatch_inbox_60 (lines 0-899 / 900-949 /
    950-999); the checkpoint with the best VALIDATION makespan is kept.

This replaces the original train.py, which rewarded schedules after 1000 local-search
steps, used a per-instance moving-average baseline, saved on the noisy training
average, and needed ~15 s per epoch on a CPU (10,000 epochs ~ 40 h).

Usage (from Static_alogorithm/GNN):
  python train.py --minutes 60                                     # -> gnn_scheduler_extend.pth
  python train.py --minutes 30 --init gnn_scheduler_extend.pth     # continue training
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from GNN import DEFAULT_WEIGHTS, GNN_NUM_THREADS, SchedulerGNN, build_schedules  # noqa: E402
from GA.GA import decode_schedule_tick_by_tick, load_dispatch_events  # noqa: E402


def real_makespan(ind, jobs) -> float:
    return max(decode_schedule_tick_by_tick(ind, jobs, check_collision=True)[0].values())


def evaluate(model, events) -> float:
    with torch.no_grad():
        inds, _ = build_schedules([e["jobs"] for e in events], model, deterministic=True)
    return sum(real_makespan(i, e["jobs"]) for i, e in zip(inds, events)) / len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default="../../test_case/static/dispatch_inbox_60.jsonl")
    ap.add_argument("--minutes", type=float, default=60.0, help="wall-clock training budget")
    ap.add_argument("--batch", type=int, default=16, help="schedules sampled per instance")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--init", default="", help="optional weights to start from")
    ap.add_argument("--out", default=DEFAULT_WEIGHTS.name)
    ap.add_argument("--log", default="train_log.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(GNN_NUM_THREADS)  # tiny tensors: more threads only add overhead
    events = load_dispatch_events(Path(args.inbox))
    train, val = events[:900], events[900:950]
    print(f"train {len(train)}  val {len(val)}  (test = lines 950-999, not used here)")

    model = SchedulerGNN()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = evaluate(model, val)
    torch.save(model.state_dict(), args.out)
    print(f"epoch 0  val {best_val:.1f}  (saved)")
    with open(args.log, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "minutes", "train_mean_makespan", "val_makespan", "best_val"])
        csv.writer(f).writerow([0, 0, "", f"{best_val:.2f}", f"{best_val:.2f}"])

    start = time.time(); epoch = 0; recent = []
    while (time.time() - start) / 60 < args.minutes:
        epoch += 1
        ev = random.choice(train)
        inds, logp = build_schedules([ev["jobs"]] * args.batch, model, deterministic=False)
        ms = torch.tensor([real_makespan(i, ev["jobs"]) for i in inds])
        adv = (ms.mean() - ms) / (ms.std() + 1e-6)
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        recent.append(ms.mean().item())

        if epoch % args.val_every == 0:
            v = evaluate(model, val)
            saved = v < best_val
            if saved:
                best_val = v
                torch.save(model.state_dict(), args.out)
            mins = (time.time() - start) / 60
            tm = sum(recent) / len(recent); recent = []
            print(f"epoch {epoch}  {mins:5.1f} min  train {tm:.1f}  val {v:.1f}  best {best_val:.1f}{'  (saved)' if saved else ''}", flush=True)
            with open(args.log, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{mins:.2f}", f"{tm:.2f}", f"{v:.2f}", f"{best_val:.2f}"])
    print(f"done: {epoch} epochs, best validation makespan {best_val:.1f} -> {args.out}")


if __name__ == "__main__":
    main()
