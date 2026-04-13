import GNN_DDQN_V7 as module
env = module.GridEnv()
env.ep_idx = 0
state = env.reset()
print("Initial mask:", env.get_action_mask())
print("Simulation time:", env.sim_time)
print("Arrival version:", env.arrival_version)
print("Last resched version:", getattr(env, "last_resched_version", None))

print("\nExecuting Wait (0) for 10 steps...")
for _ in range(10):
    env.step(0)
    print(f"t={env.sim_time}, mask={env.get_action_mask()}, arr_v={env.arrival_version}")

print("\nExecuting Reschedule (1)")
env.step(1)
print(f"t={env.sim_time}, mask={env.get_action_mask()}, arr_v={env.arrival_version}, last_resched_v={env.last_resched_version}")
