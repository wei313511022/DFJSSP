# Moves unused DFJSSP files into _unused\ (keeping their original relative paths).
# Never overwrites; writes a log to _unused\MOVED.txt.
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "_unused"
$items = @(
  # --- A: nothing imports or loads these ---
  "Random_Job_Arrivals\models\GNN_DDQN_GA_Optimizer.py",
  "Random_Job_Arrivals\models\GNN_DDQN_GA_V2.py",
  "Random_Job_Arrivals\models\GNN_DDQN_GA_V3.py",
  "Random_Job_Arrivals\models\GNN_DDQN_GA_V4.py",
  "Random_Job_Arrivals\models\GNN_DDQN_GA_V5.py",
  "Random_Job_Arrivals\models\GNN_DDQN_V6.py",
  "Random_Job_Arrivals\models\GNN_DDQN_scheduler.py",
  "Random_Job_Arrivals\models\GNN_DDQN_scheduler_Nearest.py",
  "Random_Job_Arrivals\models_pth\gnn_ddqn_model_v2",
  "Random_Job_Arrivals\models_pth\gnn_ddqn_model_v3",
  "Random_Job_Arrivals\models_pth\gnn_ddqn_model_v4",
  "Random_Job_Arrivals\models_pth\gnn_ddqn_model_v5",
  "Random_Job_Arrivals\models_pth\gnn_ddqn_model_v6",
  "Static_alogorithm\GA\GA_with_collisions.py",
  "Static_alogorithm\GA\GA_with_collisions_summary_results.csv",
  "Static_alogorithm\GNN\gnn_scheduler_best_v1.pth",
  "schedule_outbox.jsonl",
  "Random_Job_Arrivals\schedule_outbox.jsonl",
  ".vscode",
  "AMRs_Routing\__pycache__",
  "Random_Job_Arrivals\__pycache__",
  "Random_Job_Arrivals\models\__pycache__",
  "Static_alogorithm\__pycache__",
  "Static_alogorithm\GA\__pycache__",
  "Static_alogorithm\GNN\__pycache__",
  # --- B: standalone / one-off, not used by the current pipeline ---
  "Random_Job_Arrivals\models\GNN_DDQN_GA_V7.py",
  "Static_alogorithm\GA\GA_Routing_Output.py",
  "Static_alogorithm\GA\amr_routing.jsonl",
  "Static_alogorithm\GA\amr_routing_0.jsonl",
  "Static_alogorithm\GA\amr_schedule.jsonl",
  "Static_alogorithm\GA\amr_schedule_0.jsonl",
  "Static_alogorithm\MILP",
  "Random_Job_Arrivals\Testing_Tools\plot_intervals.py",
  "Random_Job_Arrivals\Testing_Tools\analyze_job_types.py",
  "Random_Job_Arrivals\Testing_Tools\fix.py",
  "Random_Job_Arrivals\Testing_Tools\fix_period_comparison.csv",
  "Random_Job_Arrivals\Testing_Tools\fix_period_comparison_flow.png",
  "test_case\dynamic\test_dataset_r2.jsonl",
  "test_case\dynamic\test_dataset_r4.jsonl",
  "test_case\dynamic\training_dataset_r1.jsonl",
  "test_case\dynamic\training_dataset_r2.jsonl",
  "test_case\dynamic\training_dataset_r4.jsonl"
)
$log = Join-Path $dest "MOVED.txt"
"Moved on $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Out-File $log -Encoding utf8
$ok = 0; $fail = 0
foreach ($rel in $items) {
  $src = Join-Path $root $rel
  $dst = Join-Path $dest $rel
  if (-not (Test-Path -LiteralPath $src)) { "SKIP (not found)  $rel" | Tee-Object -FilePath $log -Append; continue }
  if (Test-Path -LiteralPath $dst)       { "SKIP (exists)     $rel" | Tee-Object -FilePath $log -Append; $fail++; continue }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
  try { Move-Item -LiteralPath $src -Destination $dst -ErrorAction Stop; "MOVED  $rel" | Tee-Object -FilePath $log -Append; $ok++ }
  catch { "FAIL   $rel : $($_.Exception.Message)" | Tee-Object -FilePath $log -Append; $fail++ }
}
"DONE moved=$ok failed_or_skipped=$fail" | Tee-Object -FilePath $log -Append
