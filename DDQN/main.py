import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim

from env import TaskSchedulingEnv
from evaluator import print_batch_results, run_test_and_plot
from model import QNetwork
from trainer import prepare_scenarios, train_ddqn


def main():
    # Data source and generation
    task_file = "dispatch_batches.jsonl"
    train_data_dir = os.path.join(os.getcwd(), "train_data")
    auto_generate_data = True
    gen_batches = 15
    gen_size = None
    gen_min_size = 1
    gen_max_size = 25
    gen_arrival_mean = 100
    gen_seed = None
    multi_streams = True
    num_streams = 200
    base_seed = 100
    stream_file_template = "dispatch_batches_{i}.jsonl"

    # Scenario sampling
    sampling_mode = "full"  # full | window | subset
    window_size = 10
    subset_size = 10

    # Train / test switches
    do_train = True
    do_test = True

    # Model checkpoint
    save_model_path = "ddqn_policy.pt"
    load_model_path = None

    # Optimization and DDQN
    state_dim = 28
    action_dim = 4
    input_dim = state_dim + action_dim
    lr = 1e-3
    num_episodes = 30
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    allow_proactive_replenish = True

    # Training visualization and profiling
    show_train_schedule = True
    train_schedule_every_episodes = 1
    train_schedule_every_steps = 10
    train_schedule_window = 120.0
    train_schedule_window_all_axes = False
    train_schedule_pause = 0.01
    train_schedule_show_labels = False
    train_schedule_figsize = (14, 8)
    show_train_route_map = True
    train_route_map_every_episodes = 1
    train_route_map_every_steps = 1
    train_route_map_pause = 0.01
    train_route_map_figsize = (9, 8)
    train_route_map_animate = True
    train_route_map_time_step = 0.5
    train_route_map_max_frames_per_update = 120
    train_route_map_delay_seconds = 20.0
    enable_profile = True
    profile_cuda_sync = True

    # Test and plotting
    test_scenario_file = "test_scenario_one_time.jsonl"
    show_live = False
    show_live_stream = False
    show_interactive = False
    show_route_map = True
    show_plotly = True
    show_test_plots = True
    live_pause = 0.05
    live_job_file = "live_jobs.jsonl"
    live_start_at_end = True
    live_poll_interval = 0.5
    live_idle_sleep = 0.1
    live_max_steps = 100
    live_max_sim_time = None
    live_init_scenario = []
    live_record_dir = None
    live_record_every = 5
    live_record_dpi = 140
    live_make_gif = False
    live_gif_path = "live_schedule.gif"
    route_play_step = 0.5
    route_play_interval_ms = 120

    scenario_list = prepare_scenarios(
        task_file=task_file,
        train_data_dir=train_data_dir,
        auto_generate_data=auto_generate_data,
        gen_batches=gen_batches,
        gen_size=gen_size,
        gen_min_size=gen_min_size,
        gen_max_size=gen_max_size,
        gen_arrival_mean=gen_arrival_mean,
        gen_seed=gen_seed,
        multi_streams=multi_streams,
        num_streams=num_streams,
        base_seed=base_seed,
        stream_file_template=stream_file_template,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    env = TaskSchedulingEnv()
    env.allow_proactive_replenish = allow_proactive_replenish
    policy_net = QNetwork(input_dim).to(device)
    target_net = QNetwork(input_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if load_model_path and os.path.exists(load_model_path):
        ckpt = torch.load(load_model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            policy_net.load_state_dict(ckpt["model_state"])
            if "target_state" in ckpt:
                target_net.load_state_dict(ckpt["target_state"])
            else:
                target_net.load_state_dict(policy_net.state_dict())
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            policy_net.load_state_dict(ckpt)
            target_net.load_state_dict(policy_net.state_dict())
        print(f"Loaded model from {load_model_path}")

    if do_train:
        train_ddqn(
            env=env,
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            criterion=criterion,
            scenario_list=scenario_list,
            device=device,
            num_episodes=num_episodes,
            batch_size=batch_size,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            sampling_mode=sampling_mode,
            window_size=window_size,
            subset_size=subset_size,
            show_train_schedule=show_train_schedule,
            train_schedule_every_episodes=train_schedule_every_episodes,
            train_schedule_every_steps=train_schedule_every_steps,
            train_schedule_window=train_schedule_window,
            train_schedule_window_all_axes=train_schedule_window_all_axes,
            train_schedule_pause=train_schedule_pause,
            train_schedule_show_labels=train_schedule_show_labels,
            train_schedule_figsize=train_schedule_figsize,
            show_train_route_map=show_train_route_map,
            train_route_map_every_episodes=train_route_map_every_episodes,
            train_route_map_every_steps=train_route_map_every_steps,
            train_route_map_pause=train_route_map_pause,
            train_route_map_figsize=train_route_map_figsize,
            train_route_map_animate=train_route_map_animate,
            train_route_map_time_step=train_route_map_time_step,
            train_route_map_max_frames_per_update=train_route_map_max_frames_per_update,
            train_route_map_delay_seconds=train_route_map_delay_seconds,
            enable_profile=enable_profile,
            profile_cuda_sync=profile_cuda_sync,
        )

        if save_model_path:
            ckpt = {
                "model_state": policy_net.state_dict(),
                "target_state": target_net.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "input_dim": input_dim,
            }
            torch.save(ckpt, save_model_path)
            print(f"Saved model to {save_model_path}")

    if do_test:
        run_test_and_plot(
            env=env,
            policy_net=policy_net,
            device=device,
            scenario_list=scenario_list,
            test_scenario_file=test_scenario_file,
            show_live=show_live,
            show_live_stream=show_live_stream,
            show_interactive=show_interactive,
            show_route_map=show_route_map,
            show_plotly=show_plotly,
            show_test_plots=show_test_plots,
            live_pause=live_pause,
            live_job_file=live_job_file,
            live_start_at_end=live_start_at_end,
            live_poll_interval=live_poll_interval,
            live_idle_sleep=live_idle_sleep,
            live_max_steps=live_max_steps,
            live_max_sim_time=live_max_sim_time,
            live_init_scenario=live_init_scenario,
            live_record_dir=live_record_dir,
            live_record_every=live_record_every,
            live_record_dpi=live_record_dpi,
            live_make_gif=live_make_gif,
            live_gif_path=live_gif_path,
            route_play_step=route_play_step,
            route_play_interval_ms=route_play_interval_ms,
        )
        print_batch_results(env=env, policy_net=policy_net, device=device, scenario_list=scenario_list)


if __name__ == "__main__":
    main()
