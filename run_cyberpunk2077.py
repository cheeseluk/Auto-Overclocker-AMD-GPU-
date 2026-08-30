import os
import shutil
import glob
import json
import subprocess
import time

def run_cyberpunk2077():
    # --- CONFIGURATION PATHS ---

    with open(r"config.json", "r") as f:
          config = json.load(f)
 
    executable = os.path.expandvars(config["paths"]["cp2077_exe"])
    working_dir = os.path.expandvars(config["paths"]["cp2077_dir"])
    
    # Standard Windows AppData path for Cyberpunk local telemetry files
    bench_dir = os.path.expandvars(config["paths"]["bench_dir"])

    # --- 1. PURGE OLD LOGS ---
    print("Clearing out stale benchmark logs...")
    if os.path.exists(bench_dir):
        try:
            shutil.rmtree(bench_dir)
        except Exception as e:
            print(f"Warning: Could not clear benchmark directory due to lock file: {e}")
    os.makedirs(bench_dir, exist_ok=True)

    # --- 2. EXECUTE THE BENCHMARK ---
    # Notes: True CLI syntax uses flags without values for flags like -benchmark and -bNoUserProfiles
    args = [
        executable,
        "-benchmark",
        "-preset", "Ultra",
        "-width", "3840",
        "-height", "2160",
        "-fullscreen",
        "-bNoUserProfiles"
    ]
    
    print(f"Starting Cyberpunk 2077 at 4K Ultra...")
    
    try:
        # We enforce a 4-minute absolute hardware timeout. 
        # If your overclock locks the GPU core up, Python won't hang infinitely.
        subprocess.run(args, cwd=working_dir, check=True, timeout=240)
        
    except subprocess.CalledProcessError as e:
        print(f"Overclock Failure: Cyberpunk 2077 crashed with exit code ({e.returncode}).")
        return (0.0, 0.0)
    except subprocess.TimeoutExpired:
        print("Overclock Failure: Engine hung up or frozen during rendering pass (Timeout).")
        return (0.0, 0.0)
    except FileNotFoundError:
        print(f"Configuration Error: '{executable}' was not found.")
        return (0.0, 0.0)

    # --- 3. SAFETY THERMAL & MEMORY FLUSH COOLDOWN ---
    print("Benchmark complete. Process terminated cleanly.")
    print("Waiting 10 seconds for VRAM flush and thermal cooling before parsing...")
    time.sleep(10)

    # --- 4. FETCH AND READ SUMMARY.JSON ---
    # Using recursive globbing to find the single summary file inside the newly created timestamp folder
    json_search_pattern = os.path.join(bench_dir, "**", "summary.json")
    found_files = glob.glob(json_search_pattern, recursive=True)

    if not found_files:
        print("Error: Cyberpunk finished but no 'summary.json' telemetry file was created.")
        return 0.0

    target_json_path = found_files[0]
    print(f"Found runtime telemetry file at: {target_json_path}")

    try:
        with open(target_json_path, 'r', encoding='utf-8') as f:
            telemetry_data = json.load(f)
            metrics = telemetry_data["Data"]
            
        avg_fps = metrics["averageFps"]
        min_fps = metrics["minFps"]
        
        print(f"Performance Captured -> Avg: {avg_fps:.2f} FPS | Min: {min_fps:.2f} FPS")

        # --- 5. CALCULATE PENALIZED STABILITY SCORE ---
        score = avg_fps - ((avg_fps - min_fps) / 2.0)
        print(f"Calculated Optimization Metric Score: {score:.4f}")
        return (avg_fps, min_fps)

    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error reading or parsing target metrics out of telemetry file: {e}")
        return 0.0

if __name__ == "__main__":
    # Test execution execution trace
    trial_score = run_cyberpunk2077()
    print(f"\nFinal Returned Script Score: {trial_score}")