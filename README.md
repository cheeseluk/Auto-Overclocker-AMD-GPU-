# Automatic GPU Overclocker (AMD GPU)
An automation framework that uses Bayesian optimization (Optuna) and AMD's native C++ API to find stable GPU overclocking and undervolting settings.



## Demo

<img width="1045" height="880" alt="optimization_history" src="https://github.com/user-attachments/assets/8d365d1f-1056-4f1c-b0d6-0e4b580a4f8e" />

50 trials of the Optuna sampler searching the clock/voltage space (note that the 0th trial is a default configuration/baseline run). The gif shows every trial's avg/min FPS (faint dots), a red step-line tracking the best-so-far trial by composite score, and red X's marking trials that crashed the system before producing a benchmark. The overclock

[![Watch the video](https://img.youtube.com/vi/YyaOEYRJmVg/maxresdefault.jpg)](https://youtu.be/YyaOEYRJmVg)
A video demo of the program recovering fr<img width="2720" height="2440" alt="gpu_overclocker_architecture" src="https://github.com/user-attachments/assets/b73d1622-fcdf-4430-bf54-ef455f6adb6f" />
om a crash, setting gpu parameters, running a benchmark, logging results, and deciding the next test gpu parameters. 

---

## Results

| Metric | Default / Stock | Tuned (Best Trial) | Δ |
|---|---|---|---|
| Avg FPS | 71.2 | 75.6 | +4.3 (+6.0%) |
| Min FPS | 60.3 | 65.0 | +4.7 (+7.8%) |
| Voltage Offset | 0 mV | -75 mV | -75 mV |
| Power Draw | 303 W | 321 W | +18 W |
| Trials Run | — | 50 | — |
| Crashes / BSODs Recovered | — | 7 | — |

<img width="990" height="715" alt="pareto_front" src="https://github.com/user-attachments/assets/93606159-058b-455b-b872-c04a08c83c5f" />
The Pareto frontier evolves as trials accumulate — new configurations join the front and displace previously "best" trade-offs, while the overall best trial (gold star) only ever holds steady or improves, climbing from a score*[remember to finish the apsotrophe] of 65.72 to 70.34 by trial 51.


## How It Works

The system separates concerns into a high-level ML controller and a low-level hardware interface, connected by a persistent, crash-resilient execution loop:

1. **Suggest** — `brain.py` (Optuna, multivariate TPE sampler) proposes a candidate parameter set (core clock, memory clock, memory timing, voltage offset, power limits).
2. **Apply** — `AutoOverclocker.exe` (compiled from `hardware_set.cpp`, AMD ADLX SDK) injects the parameters directly into the GPU driver.
3. **Benchmark** — a Python helper launches a Cyberpunk 2077 benchmark pass and captures FPS metrics.
4. **Log & Iterate** — results (or a recorded crash) are written back to the Optuna study; the model updates and proposes the next candidate.

<img width="2720" height="2440" alt="gpu_overclocker_architecture" src="https://github.com/user-attachments/assets/30e8f216-9318-4b07-b039-eae6de7c894d" />


**Why multivariate TPE?** Core clock, memory clock, and voltage offset aren't independent — a given voltage offset is only stable up to a certain clock speed, and vice versa. A multivariate sampler models that joint relationship instead of exploring each axis independently, so it converges faster and avoids proposing configurations that look fine on each axis alone but sit in a hardware "dead zone" together.

**Why Pareto / multi-objective?** The goal isn't the single fastest configuration — it's the smallest voltage undervolt that's still stable at a good clock speed (lower voltage = less heat, less power draw, longer hardware life at the same performance). A Pareto/multi-objective study lets the optimizer report the whole frontier of performance vs. |voltage offset| trade-offs rather than collapsing to one number.

---

## Tech Stack

- **Optimization / Controller:** Python, [Optuna](https://optuna.org/) (multivariate TPE sampler, Pareto/multi-objective study)
- **Hardware Interface:** C++, [AMD ADLX SDK](https://gpuopen.com/adlx/)
- **Benchmark Harness:** Python, Cyberpunk 2077 automated benchmark capture
- **Persistence / Crash Recovery:** `state.json` state machine, SQLite (Optuna `sqlite:///overclock_study.db` storage — trial-by-trial writes mean the study survives a BSOD even if the live dashboard capture doesn't), Windows batch script for reboot-resume
- **Platform:** Windows, AMD RDNA GPUs
---

## Setup & Build

### Prerequisites
### Prerequisites
- **OS:** Windows 10/11 x64
- **GPU:** Supported AMD RDNA Discrete GPU with active AMD Adrenalin drivers.
- **C++ Tools:** Visual Studio with the "Desktop development with C++" workload installed.
- **AMD SDK:** Download the [AMD ADLX SDK](https://gpuopen.com/adlx/) and link the includes to your C++ environment.
- **Python:** Python 3.10+ (Install dependencies via `pip install requirements.txt`).
### Build the hardware interface (C++)
```bash
[TODO: build commands for mainManualPowerTuning.cpp -> AutoOverclocker.exe]
```

### Run the controller
```bash
[TODO]
```

### Resuming after a crash
[TODO: describe running the batch script on boot / how the loop auto-resumes from state.json]

### Building the history GIF
## Safety & Crash Resilience

⚠️ **This tool intentionally pushes GPU hardware past its stable operating envelope.** System freezes and BSODs are an expected, designed-for part of the search process, not a bug.

To make that safe(r) and self-recovering, the project includes:

- **Pending-state journaling** — before any parameter set is applied, it's written to `state.json` as `pending`. Nothing is marked successful until a benchmark pass actually completes.
- **Crash-aware resume** — if a BSOD occurs mid-trial, a batch script relaunches the loop on Windows startup. A watchdog routine reads the `pending` entry left behind by the crashed trial.
- **Automatic penalty scoring** — that fatal configuration is fed back into the Optuna study with a severe penalty (-10000), steering the optimizer away from that region of the search space instead of retrying it blindly.
- **Driver-safe teardown** — the ADLX wrapper includes deallocation workarounds to avoid memory access violations in the AMD driver DLL on thread termination, reducing crash frequency from the tooling itself (as opposed to the hardware limits being tested).
- **Sandbox Limits** - You can manually set the sample space by editing config.json.

**Use at your own risk.** Overclocking/undervolting can cause system instability, data loss from unexpected reboots, and — in rare cases — hardware damage. Not affiliated with or endorsed by AMD, software is provided as is, test on hardware you're WILLING TO RISK. No warranty of any kind, express or implied. 

---
