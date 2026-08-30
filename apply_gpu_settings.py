import subprocess
import sys
import json
import os

with open(r"config.json", "r") as f:
          config = json.load(f)

CPP_EXECUTABLE_PATH = os.path.expandvars(config["paths"]["cpp_executable"])

def apply_gpu_settings(trial_num, params):
    """Calls the C++ executable to apply hardware settings via ADLX."""
    print(f"\n[Trial {trial_num}] Applying Settings: {params}")
        
    # Order changed here to match: <max_gpu_freq> <mem_freq> <mem_timing> <power_limit> <voltage>
    cmd_args = [
        CPP_EXECUTABLE_PATH,
        str(params["freq_gpu"]),
        str(params["memfreq"]),
        str(params["mem_timing"]),
        str(params["power_limit"]),
        str(params["voltage"])
    ]
    
    try:
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True
        )
        
        # Print outputs from C++ to help with debugging
        if result.stdout:
            print(f"[C++ STDOUT]:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"[C++ STDERR]:\n{result.stderr.strip()}")
            
     # Check if the C++ program triggered a crash internally
        # We check both returncode and look for the successful message string
        if result.returncode != 0 and "[SUCCESS]" not in result.stdout:
            print(f"[Trial {trial_num}] 💥 C++ REPORTED INSTABILITY/CRASH DURING APPLICATION!")
            return False
            
        print(f"[Trial {trial_num}] Settings applied successfully.")
        return True
        
    except FileNotFoundError:
        print(f"ERROR: Cannot find C++ executable at {CPP_EXECUTABLE_PATH}")
        sys.exit(1)


