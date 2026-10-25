"""Train one of the DDQN reschedulers with a fixed seed, episode count and output folder,
without editing the model file.

Usage (from Random_Job_Arrivals):
  python -u train_ddqn.py --module GA_models.GA_DDQN --episodes 300 --seed 42 --save_path ../models_pth/ga_ddqn_model_s42
  python -u train_ddqn.py --module GNN_models.GNN_DDQN_V7 ...
  python -u train_ddqn.py --module Taguchi_models.Taguchi_DDQN ...

--save_path is relative to the module's folder (like the CONFIG paths), so
'../models_pth/x' lands in Random_Job_Arrivals/models_pth/x.
"""
import argparse
import importlib
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="e.g. GA_models.GA_DDQN")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads; the networks are tiny, more threads only add overhead")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override CONFIG entries, e.g. DATASET_PATH=../../test_case/dynamic/training_dataset_r4.jsonl SIM_TIME=500")
    args = ap.parse_args()

    folder, name = args.module.rsplit(".", 1)
    module_dir = os.path.join(HERE, *folder.split("."))
    os.chdir(module_dir)                  # CONFIG paths are relative to the module folder
    sys.path.insert(0, module_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    mod = importlib.import_module(name)
    mod.CONFIG["NUM_EPISODES"] = args.episodes
    mod.CONFIG["SAVE_PATH"] = args.save_path
    for kv in args.set:
        key, value = kv.split("=", 1)
        if key not in mod.CONFIG:
            raise SystemExit(f"unknown CONFIG key: {key}")
        old = mod.CONFIG[key]
        mod.CONFIG[key] = type(old)(value) if isinstance(old, (int, float)) and not isinstance(old, bool) else value
    os.makedirs(args.save_path, exist_ok=True)
    print(f"module {args.module} | episodes {args.episodes} | seed {args.seed} | save {os.path.abspath(args.save_path)}"
          f" | dataset {mod.CONFIG['DATASET_PATH']} | SIM_TIME {mod.CONFIG['SIM_TIME']}", flush=True)
    mod.main()


if __name__ == "__main__":
    main()
