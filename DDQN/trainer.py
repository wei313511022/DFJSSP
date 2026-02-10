import os
import random
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_io import load_records
from env import TaskSchedulingEnv
from features import (
    build_actions_for_tasks,
    flatten_jobs,
    q_values_batch,
)
from viz_matplotlib import draw_amr_schedule, draw_dispatch_queue, draw_input_queue
from viz_route_map import draw_route_map


def sample_scenario(records: List[dict], mode: str, window_size: int, subset_size: int):
    n = len(records)
    if n == 0:
        return records
    if mode == "window":
        if window_size <= 0 or window_size >= n:
            return records
        start = random.randint(0, n - window_size)
        return records[start : start + window_size]
    if mode == "subset":
        k = min(subset_size, n) if subset_size > 0 else n
        chosen = random.sample(records, k)
        chosen.sort(key=lambda r: r.get("dispatch_time", 0.0))
        return chosen
    return records


def prepare_scenarios(
    task_file: str,
    train_data_dir: str,
    auto_generate_data: bool,
    gen_batches: int,
    gen_size: Union[int, None],
    gen_min_size: int,
    gen_max_size: int,
    gen_arrival_mean: float,
    gen_seed: Union[int, None],
    multi_streams: bool,
    num_streams: int,
    base_seed: Union[int, None],
    stream_file_template: str,
) -> List[Union[dict, List[dict]]]:
    def collect_stream_paths() -> List[str]:
        paths: List[str] = []

        # Prefer glob discovery to tolerate missing indices (e.g., 0,1,3,...).
        if train_data_dir:
            if "{i}" in stream_file_template:
                glob_pattern = stream_file_template.replace("{i}", "*")
            else:
                glob_pattern = stream_file_template
            discovered = sorted(Path(train_data_dir).glob(glob_pattern))
            for p in discovered:
                if p.is_file() and p.suffix.lower() == ".jsonl":
                    paths.append(str(p))

        # Fallback: explicit indexed paths.
        if not paths:
            for i in range(num_streams):
                fname = stream_file_template.format(i=i)
                fpath = os.path.join(train_data_dir, fname) if train_data_dir else fname
                if os.path.exists(fpath):
                    paths.append(fpath)

        return paths

    if train_data_dir:
        os.makedirs(train_data_dir, exist_ok=True)

    generated_paths: List[str] = []
    if auto_generate_data:
        import random_job_gen as rjg

        if multi_streams:
            # When auto-generation is enabled, refresh matching stream files so
            # old data does not silently override current generation settings.
            if train_data_dir:
                if "{i}" in stream_file_template:
                    glob_pattern = stream_file_template.replace("{i}", "*")
                else:
                    glob_pattern = stream_file_template
                for p in Path(train_data_dir).glob(glob_pattern):
                    if p.is_file() and p.suffix.lower() == ".jsonl":
                        p.unlink()

            for i in range(num_streams):
                out_file = (
                    os.path.join(train_data_dir, stream_file_template.format(i=i))
                    if train_data_dir
                    else stream_file_template.format(i=i)
                )
                rjg.generate_data(
                    num_batches=gen_batches,
                    fixed_size=gen_size,
                    min_size=gen_min_size,
                    max_size=gen_max_size,
                    arrival_mean=gen_arrival_mean,
                    output_file=out_file,
                    station_count=5,
                    seed=(base_seed + i) if base_seed is not None else None,
                )
                generated_paths.append(out_file)
        else:
            out_file = os.path.join(train_data_dir, task_file) if train_data_dir else task_file
            rjg.generate_data(
                num_batches=gen_batches,
                fixed_size=gen_size,
                min_size=gen_min_size,
                max_size=gen_max_size,
                arrival_mean=gen_arrival_mean,
                output_file=out_file,
                station_count=5,
                seed=gen_seed,
            )
            generated_paths.append(out_file)

    if multi_streams:
        scenario_list = []
        if generated_paths:
            stream_paths = [p for p in generated_paths if os.path.exists(p)]
        else:
            stream_paths = collect_stream_paths()
        for fpath in stream_paths:
            recs = load_records(fpath)
            if recs:
                scenario_list.append(recs)
        if not scenario_list:
            raise RuntimeError(
                "No records found in streams "
                f"(dir={train_data_dir}, template={stream_file_template})"
            )
    else:
        if generated_paths:
            path = generated_paths[0]
        else:
            path = os.path.join(train_data_dir, task_file) if train_data_dir else task_file
        records = load_records(path)
        if not records:
            raise RuntimeError(f"No records found in {task_file}")
        scenario_list = [records]

    return scenario_list


def train_ddqn(
    env: TaskSchedulingEnv,
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scenario_list: List[Union[dict, List[dict]]],
    device: torch.device,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    epsilon: float,
    epsilon_end: float,
    epsilon_decay: float,
    sampling_mode: str,
    window_size: int,
    subset_size: int,
    show_train_schedule: bool,
    train_schedule_every_episodes: int,
    train_schedule_every_steps: int,
    train_schedule_window: float,
    train_schedule_window_all_axes: bool,
    train_schedule_pause: float,
    train_schedule_show_labels: bool,
    train_schedule_figsize: tuple,
    show_train_route_map: bool,
    train_route_map_every_episodes: int,
    train_route_map_every_steps: int,
    train_route_map_pause: float,
    train_route_map_figsize: tuple,
    train_route_map_animate: bool,
    train_route_map_time_step: float,
    train_route_map_max_frames_per_update: int,
    train_route_map_delay_seconds: float,
    enable_profile: bool,
    profile_cuda_sync: bool,
) -> Dict[str, Any]:
    experience_t = namedtuple(
        "Experience", ["state", "a_feat", "reward", "next_state", "next_feats", "done"]
    )
    memory = deque(maxlen=50000)

    input_dim = next(policy_net.parameters()).shape[1]
    makespans: List[float] = []

    plt.ion()
    fig_train, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_mk = axes[0, 0]
    ax_loss = axes[0, 1]
    ax_eps = axes[1, 0]
    ax_dummy = axes[1, 1]
    ax_dummy_right = ax_dummy.twinx()

    mk_history: List[float] = []
    mk_ma50_history: List[float] = []
    loss_history: List[float] = []
    loss_ma100_history: List[float] = []
    eps_history: List[float] = []
    mk_per_job_history: List[float] = []
    mk_ratio_history: List[float] = []
    mk_per_job_ma50: List[float] = []
    mk_ratio_ma50: List[float] = []

    prof: Dict[str, float] = {}

    def prof_add(key: str, dt: float) -> None:
        prof[key] = prof.get(key, 0.0) + dt

    def sync_cuda() -> None:
        if profile_cuda_sync and device.type == "cuda":
            torch.cuda.synchronize()

    train_sched_fig = None
    train_sched_axes = None
    train_route_fig = None
    train_route_ax = None
    train_route_text = None

    def moving_avg(x: List[float], w: int) -> float:
        if len(x) == 0:
            return float("nan")
        if len(x) < w:
            return float(sum(x) / len(x))
        return float(sum(x[-w:]) / w)

    def set_auto_ylim(
        ax,
        series_list: List[List[float]],
        margin_ratio: float = 0.08,
        min_span: float = 1e-6,
    ) -> None:
        vals: List[float] = []
        for s in series_list:
            if not s:
                continue
            vals.extend([float(v) for v in s if np.isfinite(v)])
        if not vals:
            return

        y_min = float(min(vals))
        y_max = float(max(vals))
        span = y_max - y_min
        if span < min_span:
            center = 0.5 * (y_min + y_max)
            base = max(abs(center), 1.0)
            pad = max(min_span, base * margin_ratio)
            ax.set_ylim(center - pad, center + pad)
            return

        pad = span * margin_ratio
        ax.set_ylim(y_min - pad, y_max + pad)

    def update_train_plot(ep: int):
        ax_mk.clear()
        ax_mk.plot(mk_history, linewidth=1, label="makespan")
        ax_mk.plot(mk_ma50_history, linewidth=2, label="MA-50")
        ax_mk.set_title("Makespan per episode (lower is better)")
        ax_mk.set_xlabel("Episode")
        ax_mk.set_ylabel("Makespan")
        ax_mk.grid(True, linestyle="--", alpha=0.4)
        set_auto_ylim(ax_mk, [mk_history, mk_ma50_history])
        ax_mk.legend(loc="upper right")

        ax_loss.clear()
        if len(loss_history) > 0:
            ax_loss.plot(loss_history, linewidth=1, label="loss")
            ax_loss.plot(loss_ma100_history, linewidth=2, label="MA-100")
            set_auto_ylim(ax_loss, [loss_history, loss_ma100_history])
            ax_loss.legend(loc="upper right")
        ax_loss.set_title("Training loss (TD error)")
        ax_loss.set_xlabel("Update step")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.4)

        ax_eps.clear()
        ax_eps.plot(eps_history, linewidth=2)
        ax_eps.set_title("Epsilon (exploration rate)")
        ax_eps.set_xlabel("Episode")
        ax_eps.set_ylabel("epsilon")
        ax_eps.set_ylim(0.0, 1.05)
        ax_eps.grid(True, linestyle="--", alpha=0.4)

        ax_dummy.clear()
        ax_dummy_right.clear()
        left_lines = []
        right_lines = []
        if len(mk_per_job_history) > 0:
            left_lines.extend(
                ax_dummy.plot(mk_per_job_history, linewidth=1, color="tab:blue", label="mk/job")
            )
            left_lines.extend(
                ax_dummy.plot(mk_per_job_ma50, linewidth=2, color="tab:orange", label="mk/job MA-50")
            )
            set_auto_ylim(ax_dummy, [mk_per_job_history, mk_per_job_ma50])
        if len(mk_ratio_history) > 0:
            right_lines.extend(
                ax_dummy_right.plot(mk_ratio_history, linewidth=1, color="tab:green", label="mk/proc")
            )
            right_lines.extend(
                ax_dummy_right.plot(mk_ratio_ma50, linewidth=2, color="tab:red", label="mk/proc MA-50")
            )
            set_auto_ylim(ax_dummy_right, [mk_ratio_history, mk_ratio_ma50])
        if left_lines or right_lines:
            handles = left_lines + right_lines
            labels = [h.get_label() for h in handles]
            ax_dummy.legend(handles, labels, loc="upper right")
        ax_dummy.set_title("Normalized Makespan")
        ax_dummy.set_xlabel("Episode")
        ax_dummy.set_ylabel("mk/job")
        ax_dummy_right.set_ylabel("mk/proc")
        ax_dummy.grid(True, linestyle="--", alpha=0.4)

        fig_train.suptitle(f"Training Monitor | EP={ep+1}", fontsize=14)
        fig_train.tight_layout()
        fig_train.canvas.draw()
        fig_train.canvas.flush_events()

    train_wall_start = time.perf_counter()
    for ep in range(num_episodes):
        scenario_idx = random.randrange(len(scenario_list))
        base_scenario = scenario_list[scenario_idx]
        if sampling_mode != "full" and isinstance(base_scenario, list):
            scenario = sample_scenario(base_scenario, sampling_mode, window_size, subset_size)
        else:
            scenario = base_scenario
        is_stream = (
            isinstance(scenario, list)
            and len(scenario) > 0
            and isinstance(scenario[0], dict)
            and "jobs" in scenario[0]
        )
        scenario_tag = f"stream:{len(scenario)}" if is_stream else str(scenario_idx)
        dispatch_time = (
            float(scenario[0].get("dispatch_time", 0.0))
            if is_stream
            else float(scenario.get("dispatch_time", 0.0))
            if isinstance(scenario, dict)
            else 0.0
        )
        job_count = len(flatten_jobs(scenario))
        s = np.array(env.reset(scenario), dtype=np.float32)
        release_t0 = env.release_events[0][0] if env.release_events else float("nan")
        show_schedule_this_ep = show_train_schedule and (
            train_schedule_every_episodes <= 1 or (ep % train_schedule_every_episodes == 0)
        )
        show_route_map_this_ep = show_train_route_map and (
            train_route_map_every_episodes <= 1 or (ep % train_route_map_every_episodes == 0)
        )
        if show_schedule_this_ep and train_sched_fig is None:
            train_sched_fig, train_sched_axes = plt.subplots(3, 1, figsize=train_schedule_figsize)
            train_sched_fig.subplots_adjust(hspace=0.5, top=0.9)
        if show_route_map_this_ep and train_route_fig is None:
            train_route_fig, train_route_ax = plt.subplots(figsize=train_route_map_figsize)
            train_route_fig.subplots_adjust(bottom=0.2, top=0.92)
            train_route_text = train_route_fig.text(0.02, 0.02, "", fontsize=10, ha="left", va="bottom")
        route_last_draw_t = float(env.t)
        step = 0

        done = False
        ep_reward = 0.0

        while not done:
            rid = env.current_robot

            if enable_profile:
                t0 = time.perf_counter()
            actions = build_actions_for_tasks(
                env.available_tasks,
                env.robot_inventory[rid],
                env.capacity_per_type,
                allow_proactive_replenish=env.allow_proactive_replenish,
            )
            k = len(actions)
            if k == 0:
                raise RuntimeError("No valid actions available during training step.")

            feats = np.zeros((k, 4), dtype=np.float32)
            for i, (task_idx, replenish) in enumerate(actions):
                task = env.available_tasks[task_idx]
                travel, wait, proc, rep = env.action_features(rid, task, replenish)
                feats[i] = (travel, wait, proc, rep)

            if enable_profile:
                prof_add("action_features", time.perf_counter() - t0)

            if random.random() < epsilon:
                a_idx = random.randrange(k)
                a_feat = feats[a_idx].copy()
            else:
                with torch.no_grad():
                    if enable_profile:
                        sync_cuda()
                        t_q = time.perf_counter()
                    q_all = q_values_batch(policy_net, s, feats, device)
                    if enable_profile:
                        sync_cuda()
                        prof_add("q_select", time.perf_counter() - t_q)
                    a_idx = int(torch.argmax(q_all).item())
                a_feat = feats[a_idx].copy()

            action = actions[a_idx]
            if enable_profile:
                t_env = time.perf_counter()
            sp, r, done = env.step(action)
            if enable_profile:
                prof_add("env_step", time.perf_counter() - t_env)
            ep_reward += r

            sp_arr = np.array(sp, dtype=np.float32) if sp is not None else None
            if not done:
                next_rid = env.current_robot
                next_actions = build_actions_for_tasks(
                    env.available_tasks,
                    env.robot_inventory[next_rid],
                    env.capacity_per_type,
                    allow_proactive_replenish=env.allow_proactive_replenish,
                )
                next_feats = np.zeros((len(next_actions), 4), dtype=np.float32)
                for i, (task_idx, replenish) in enumerate(next_actions):
                    task = env.available_tasks[task_idx]
                    travel, wait, proc, rep = env.action_features(next_rid, task, replenish)
                    next_feats[i] = (travel, wait, proc, rep)
            else:
                next_feats = np.zeros((0, 4), dtype=np.float32)

            memory.append(experience_t(s, a_feat, float(r), sp_arr, next_feats, bool(done)))
            s = sp_arr
            step += 1

            if show_schedule_this_ep and train_sched_axes is not None:
                if step % max(1, train_schedule_every_steps) == 0 or done:
                    for ax in train_sched_axes:
                        ax.clear()

                    draw_dispatch_queue(
                        train_sched_axes[0],
                        env.trace,
                        show_labels=train_schedule_show_labels,
                        current_t=env.t,
                    )
                    draw_amr_schedule(
                        train_sched_axes[1],
                        env.trace,
                        env.makespan(),
                        show_labels=train_schedule_show_labels,
                        current_t=env.t,
                        inventories=env.robot_inventory,
                    )
                    draw_input_queue(
                        train_sched_axes[2],
                        scenario,
                        show_labels=train_schedule_show_labels,
                        current_t=env.t,
                    )

                    if train_schedule_window and train_schedule_window > 0:
                        center = env.t
                        half = train_schedule_window / 2.0
                        left = max(0.0, center - half)
                        right = center + half
                        if train_schedule_window_all_axes:
                            for ax in train_sched_axes:
                                ax.set_xlim(left, right)
                        else:
                            train_sched_axes[1].set_xlim(left, right)

                    train_sched_fig.suptitle(
                        f"TRAIN t={env.t:.1f}s | ep={ep+1} step={step} | "
                        f"scenario={scenario_tag} dt={dispatch_time:.2f} jobs={job_count}"
                    )
                    train_sched_fig.canvas.draw()
                    train_sched_fig.canvas.flush_events()
                    plt.pause(train_schedule_pause)

            if show_route_map_this_ep and train_route_ax is not None and train_route_text is not None:
                if step % max(1, train_route_map_every_steps) == 0 or done:
                    t_sched = float(env.t)
                    delay_s = max(0.0, float(train_route_map_delay_seconds))
                    t_target = max(0.0, t_sched - delay_s)
                    if t_target < route_last_draw_t:
                        route_last_draw_t = t_target
                    frame_times: List[float]
                    frame_pause = train_route_map_pause
                    if train_route_map_animate and t_target > route_last_draw_t:
                        dt = max(1e-6, float(train_route_map_time_step))
                        n_frames_raw = int(np.ceil((t_target - route_last_draw_t) / dt))
                        n_frames = max(1, min(train_route_map_max_frames_per_update, n_frames_raw))
                        # If frame count is clipped, increase pause proportionally
                        # to avoid visible "fast-forward" jumps.
                        if n_frames_raw > n_frames and train_route_map_pause > 0:
                            frame_pause = train_route_map_pause * (n_frames_raw / n_frames)
                        frame_times = list(np.linspace(route_last_draw_t, t_target, n_frames + 1))[1:]
                    else:
                        frame_times = [t_target]

                    for t_frame in frame_times:
                        lines = draw_route_map(train_route_ax, env, env.trace, current_t=float(t_frame))
                        train_route_text.set_text("\n".join(lines))
                        train_route_fig.suptitle(
                            f"TRAIN Route | ep={ep+1} step={step} | "
                            f"scenario={scenario_tag} dt={dispatch_time:.2f} jobs={job_count} | "
                            f"sched_t={t_sched:.1f}s route_t={float(t_frame):.1f}s",
                            fontsize=12,
                        )
                        train_route_fig.canvas.draw()
                        train_route_fig.canvas.flush_events()
                        if frame_pause > 0:
                            plt.pause(frame_pause)
                    route_last_draw_t = t_target

            if len(memory) >= batch_size:
                if enable_profile:
                    t_upd = time.perf_counter()
                batch = random.sample(memory, batch_size)

                sa_batch = np.zeros((batch_size, input_dim), dtype=np.float32)
                y_batch = np.zeros((batch_size, 1), dtype=np.float32)

                for bi, exp in enumerate(batch):
                    s0 = exp.state
                    a0 = exp.a_feat
                    r0 = exp.reward
                    s1 = exp.next_state
                    feats1 = exp.next_feats
                    d0 = exp.done

                    if d0 or s1 is None or feats1 is None or len(feats1) == 0:
                        y = r0
                    else:
                        with torch.no_grad():
                            q_next_all = q_values_batch(policy_net, s1.astype(np.float32), feats1, device)
                            best_i = int(torch.argmax(q_next_all).item())
                            best_feat_next = feats1[best_i]

                            sa1 = np.concatenate([s1.astype(np.float32), best_feat_next], axis=0)[None, :]
                            sa1_t = torch.from_numpy(sa1).to(device)
                            q_t = target_net(sa1_t).item()
                            y = r0 + gamma * q_t

                    sa0 = np.concatenate([s0.astype(np.float32), a0.astype(np.float32)], axis=0)
                    sa_batch[bi] = sa0
                    y_batch[bi, 0] = y

                sa_t = torch.from_numpy(sa_batch).to(device)
                y_t = torch.from_numpy(y_batch).to(device)

                q_pred = policy_net(sa_t)
                loss = criterion(q_pred, y_t)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_val = float(loss.detach().cpu().item())
                loss_history.append(loss_val)
                loss_ma100_history.append(moving_avg(loss_history, 100))
                if enable_profile:
                    sync_cuda()
                    prof_add("ddqn_update", time.perf_counter() - t_upd)

        mk = -ep_reward
        makespans.append(mk)

        mk_history.append(mk)
        mk_per_job = mk / max(1, job_count)
        total_proc = sum(float(j.get("proc_time", 0.0)) for j in flatten_jobs(scenario))
        mk_ratio = mk / max(1e-6, total_proc)
        mk_per_job_history.append(mk_per_job)
        mk_ratio_history.append(mk_ratio)
        mk_per_job_ma50.append(moving_avg(mk_per_job_history, 50))
        mk_ratio_ma50.append(moving_avg(mk_ratio_history, 50))
        mk_ma50_history.append(moving_avg(mk_history, 50))
        eps_history.append(epsilon)

        if epsilon > epsilon_end:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        avg_mk = moving_avg(mk_history, 50)
        print(
            f"[EP {ep+1}] batch={scenario_tag} dispatch_time={dispatch_time:.2f} "
            f"release_t0={release_t0:.2f} jobs={job_count} MA-50 mk: {avg_mk:.2f}, "
            f"mk/job: {mk_per_job:.3f}, mk/proc: {mk_ratio:.3f}, "
            f"eps={epsilon:.3f}, last_mk={mk:.2f}"
        )
        update_train_plot(ep)

    target_net.load_state_dict(policy_net.state_dict())

    if enable_profile:
        train_wall_end = time.perf_counter()
        total = train_wall_end - train_wall_start
        accounted = sum(prof.values())
        print("\n=== PROFILING (TRAIN) ===")
        for k, v in sorted(prof.items(), key=lambda x: -x[1]):
            pct = (v / total * 100.0) if total > 0 else 0.0
            print(f"{k:>16}: {v:8.2f}s  ({pct:5.1f}%)")
        other = max(0.0, total - accounted)
        print(
            f"{'other':>16}: {other:8.2f}s  "
            f"({(other/total*100.0) if total>0 else 0.0:5.1f}%)"
        )
        print(f"{'total':>16}: {total:8.2f}s")

    return {
        "makespans": makespans,
        "mk_history": mk_history,
        "loss_history": loss_history,
        "epsilon_final": epsilon,
        "profile": prof,
    }
