import torch
import torch.optim as optim
import time
import numpy as np

# Import from the GNN script
from GNN import SchedulerGNN, solve_with_gnn
from GA_code.GA import make_jobs, describe_solution

def train(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Hyperparameters
    num_epochs = 100
    batch_size = 5 # Number of episodes (schedules) to sample before a weight update
    lr = 1e-3
    
    # Load test case if specified
    from GA_code.GA import load_dispatch_events
    dispatch_events = []
    if args.inbox:
        from pathlib import Path
        inbox_path = Path(args.inbox)
        if inbox_path.exists():
            dispatch_events = load_dispatch_events(inbox_path)
            print(f"Loaded {len(dispatch_events)} dispatch events from {args.inbox}")
    
    # Initialize Model and Optimizer
    model = SchedulerGNN(amr_in_dim=8, job_in_dim=10, hidden_dim=128, gnn_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Track the moving average of the makespan to use as a baseline for REINFORCE
    baseline_makespan = None
    alpha = 0.1 # Exponential moving average rate
    
    print("Starting REINFORCE training...")
    
    best_makespan = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        batch_makespans = []
        
        # We will train on random dispatch events
        for b in range(batch_size):
            # Generate a new random job scenario or pick from dispatch events
            if dispatch_events:
                import random
                jobs = random.choice(dispatch_events)["jobs"]
            else:
                jobs = make_jobs()
            
            # Forward pass: Sample a schedule
            ind, total_log_prob, solve_dur = solve_with_gnn(jobs, model, deterministic=False)
            
            # Simulator evaluation: get the exact makespan (No gantt plot during training)
            # check_collision=True is implicitly called inside describe_solution
            makespan, _ = describe_solution(ind, jobs, solve_time=solve_dur, show_gantt=False)
            
            batch_makespans.append(makespan)
            
            # --- Policy Gradient Update ---
            
            # Initialize baseline
            if baseline_makespan is None:
                baseline_makespan = makespan
                
            # Reward: Positive if we beat the baseline makespan, negative if we did worse
            reward = baseline_makespan - makespan
            
            # Loss = -log_prob * reward (We want to maximize reward, so minimize -reward)
            loss = -total_log_prob * reward
            
            # Accumulate loss for the batch
            loss.backward()
            epoch_loss += loss.item()
            
        # Update weights based on the batch gradients
        optimizer.step()
        
        # Update the moving average baseline
        avg_batch_makespan = sum(batch_makespans) / batch_size
        baseline_makespan = (1 - alpha) * baseline_makespan + (alpha * avg_batch_makespan)
        
        # Logging
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | Avg Makespan: {avg_batch_makespan:.2f} | Baseline: {baseline_makespan:.2f} | Total Loss: {epoch_loss:.4f}")
            
        # Optional: Save best model
        if avg_batch_makespan < best_makespan:
            best_makespan = avg_batch_makespan
            torch.save(model.state_dict(), "gnn_scheduler_best.pth")
            print(f"   -> Saved new best model (Makespan: {best_makespan:.2f})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", type=str, default="", help="Path to dispatch inbox JSONL file to train on (e.g., ../../test_case/dispatch_inbox_60.jsonl)")
    args = parser.parse_args()
    
    train(args)
