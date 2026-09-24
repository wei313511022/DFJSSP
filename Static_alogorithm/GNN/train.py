"""
Train the static scheduling GNN (SchedulerGNN) against the GA.py simulator.

Method (same network, features and schedule-building rule as GNN.py):
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
  python train.py --minutes 60 --out gnn_scheduler_retrained.pth
  python train.py --minutes 30 --init gnn_scheduler_retrained.pth   # continue training
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from GNN import SchedulerGNN, extract_state  # noqa: E402
from GA.GA import (  # noqa: E402
    AMR_KEYS, AMR_STARTS, STATIONS, SUPPLY_LOCATIONS, TYPE_DURATION, Individual,
    decode_schedule_tick_by_tick, heuristic, load_dispatch_events,
)


def solve_batch(job_lists: List[list], model: SchedulerGNN, deterministic: bool):
    """Build one schedule per entry of `job_lists` in a single batched pass.

    Identical decision rule and internal state update to GNN.solve_with_gnn (no
    init_state); all instances must have the same number of jobs.
    Returns (individuals, log_prob tensor of shape [B]).
    """
    B = len(job_lists)
    n = len(job_lists[0])
    device = next(model.parameters()).device
    pos = [{a: AMR_STARTS[a] for a in AMR_KEYS} for _ in range(B)]
    avail = [{a: 0.0 for a in AMR_KEYS} for _ in range(B)]
    st_avail = [{s: 0.0 for s in STATIONS} for _ in range(B)]
    inv = []
    for _ in range(B):
        d = {a: {m: 0 for m in TYPE_DURATION} for a in AMR_KEYS}
        d["AMR1"]["A"] = 3; d["AMR2"]["B"] = 3; d["AMR3"]["C"] = 3
        inv.append(d)
    assigned = [set() for _ in range(B)]
    orders = [[] for _ in range(B)]
    assign = [dict() for _ in range(B)]
    log_prob = torch.zeros(B, device=device)

    model.eval() if deterministic else model.train()
    for _ in range(n):
        feats = [extract_state(job_lists[b], assigned[b], pos[b], avail[b], inv[b]) for b in range(B)]
        amr_f = torch.cat([f[0] for f in feats]).to(device)
        job_f = torch.cat([f[1] for f in feats]).to(device)
        mask = torch.cat([f[2] for f in feats]).to(device)
        logits = model(amr_f, job_f, mask).reshape(B, -1)
        if deterministic:
            actions = logits.argmax(dim=1)
        else:
            lp = F.log_softmax(logits, dim=1)
            actions = torch.distributions.Categorical(logits=lp).sample()
            log_prob = log_prob + lp.gather(1, actions.unsqueeze(1)).squeeze(1)
        for b, act in enumerate(actions.tolist()):
            amr = AMR_KEYS[act // n]
            job = job_lists[b][act % n]
            orders[b].append(job.idx)
            assign[b][job.idx] = amr
            assigned[b].add(job.idx)
            # same fast approximation as solve_with_gnn
            t = avail[b][amr]; p = pos[b][amr]
            if inv[b][amr][job.type_] == 0:
                sup = SUPPLY_LOCATIONS[job.type_]
                t += heuristic(p, sup); p = sup
                inv[b][amr][job.type_] = 3
            target = STATIONS[job.station]
            t += heuristic(p, target)
            t = max(t, st_avail[b][job.station])
            t += job.duration
            inv[b][amr][job.type_] -= 1
            st_avail[b][job.station] = t
            avail[b][amr] = t
            pos[b][amr] = target
    inds = [Individual(order=orders[b], amr_assignment=[assign[b][i] for i in range(n)]) for b in range(B)]
    return inds, log_prob


def real_makespan(ind, jobs) -> float:
    return max(decode_schedule_tick_by_tick(ind, jobs, check_collision=True)[0].values())


def evaluate(model, events) -> float:
    with torch.no_grad():
        inds, _ = solve_batch([e["jobs"] for e in events], model, deterministic=True)
    return sum(real_makespan(i, e["jobs"]) for i, e in zip(inds, events)) / len(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", default="../../test_case/static/dispatch_inbox_60.jsonl")
    ap.add_argument("--minutes", type=float, default=60.0, help="wall-clock training budget")
    ap.add_argument("--batch", type=int, default=16, help="schedules sampled per instance")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--init", default="", help="optional weights to start from")
    ap.add_argument("--out", default="gnn_scheduler_retrained.pth")
    ap.add_argument("--log", default="train_log.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    events = load_dispatch_events(Path(args.inbox))
    train, val = events[:900], events[900:950]
    print(f"train {len(train)}  val {len(val)}  (test = lines 950-999, not used here)")

    model = SchedulerGNN(amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2)
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
        inds, logp = solve_batch([ev["jobs"]] * args.batch, model, deterministic=False)
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
