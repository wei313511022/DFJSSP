import os
import json
import time
import statistics
import subprocess

# Path to MILP.py and test_case directory
MILP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'MILP_no_gui.py'))
TEST_CASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_case'))

# Find all .jsonl files in test_case directory
jsonl_files = [f for f in os.listdir(TEST_CASE_DIR) if f.endswith('.jsonl')]

results = {}

for jsonl_file in jsonl_files:
    file_path = os.path.join(TEST_CASE_DIR, jsonl_file)
    makespans = []
    times = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            if not line.strip():
                continue
            # Write this line to a temp file as MILP.py expects a file input
            temp_inbox = os.path.join(TEST_CASE_DIR, 'temp_inbox.jsonl')
            with open(temp_inbox, 'w', encoding='utf-8') as temp_f:
                temp_f.write(line)
            # Run MILP.py and capture output
            start = time.time()
            try:
                proc = subprocess.run(['python3', MILP_PATH], cwd=os.path.dirname(MILP_PATH), capture_output=True, text=True, timeout=300)
                output = proc.stdout + proc.stderr
            except Exception as e:
                output = str(e)
            elapsed = time.time() - start
            # Parse makespan and solve_time from output
            makespan = None
            solve_time = None
            for l in output.splitlines():
                if '[MILP]' in l and 'makespan=' in l and 'solve_time=' in l:
                    try:
                        parts = l.split()
                        for part in parts:
                            if part.startswith('makespan='):
                                makespan = float(part.split('=')[1].replace('s',''))
                            if part.startswith('solve_time='):
                                solve_time = float(part.split('=')[1].replace('s',''))
                    except Exception:
                        pass
            if makespan is not None:
                makespans.append(makespan)
            else:
                makespans.append(float('nan'))
            if solve_time is not None:
                times.append(solve_time)
            else:
                times.append(elapsed)
    # Remove temp file
    try:
        os.remove(temp_inbox)
    except Exception:
        pass
    # Calculate statistics
    avg_makespan = statistics.mean([x for x in makespans if not (x != x)]) if makespans else float('nan')
    std_makespan = statistics.stdev([x for x in makespans if not (x != x)]) if len(makespans) > 1 else 0.0
    avg_time = statistics.mean([x for x in times if not (x != x)]) if times else float('nan')
    std_time = statistics.stdev([x for x in times if not (x != x)]) if len(times) > 1 else 0.0
    results[jsonl_file] = {
        'avg_makespan': avg_makespan,
        'std_makespan': std_makespan,
        'avg_time': avg_time,
        'std_time': std_time,
    }
    print(f"{jsonl_file}: avg_makespan={avg_makespan:.3f}, std_makespan={std_makespan:.3f}, avg_time={avg_time:.3f}, std_time={std_time:.3f}")

# Optionally, write results to a file
with open(os.path.join(TEST_CASE_DIR, 'milp_batch_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Done.")
