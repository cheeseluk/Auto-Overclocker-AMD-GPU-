# Automatic GPU Overclocker (AMD GPU)
An automation framework that uses Bayesian optimization (Optuna) and AMD's native C++ API to find stable GPU overclocking and undervolting settings.



## Demo


https://github.com/user-attachments/assets/07a02114-558f-488d-a253-9919ee771674


A video demo of the program recovering from a crash, setting gpu parameters, running a benchmark, logging results, and deciding the next test gpu parameters. [Youtube link](https://youtu.be/YyaOEYRJmVg) higher quality and longer demo.

<img width="1045" height="880" alt="optimization_history" src="https://github.com/user-attachments/assets/8d365d1f-1056-4f1c-b0d6-0e4b580a4f8e" />




50 trials of the Optuna sampler searching the clock/voltage space (note that the 0th trial is a default configuration/baseline run). The gif shows every trial's avg/min FPS (faint dots), a red step-line tracking the best-so-far trial by composite score, and red X's marking trials that crashed the system before producing a benchmark. The best overclocking parameters so far are shown at the bottom of the gif. Note that mem timing corresponds to the memory timing of the VRAM with 1 indicating fast timing and 0 indicating default timing.


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
The Pareto frontier evolves as trials accumulate — new configurations join the front and displace previously "best" trade-offs, while the overall best trial (gold star) only ever holds steady or improves, climbing from a score of 65.72 to 70.34 by trial 51.


## How It Works

The system separates concerns into a high-level ML controller and a low-level hardware interface, connected by a persistent, crash-resilient execution loop:

1. **Suggest** — `brain.py` (Optuna, multivariate TPE sampler) proposes a candidate parameter set (core clock, memory clock, memory timing, voltage offset, power limits).
2. **Apply** — `hardware_set.exe` (compiled from `hardware_set.cpp`, AMD ADLX SDK) injects the parameters directly into the GPU driver.
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
- **Persistence / Crash Recovery:** `state.json` state machine, SQLite (Optuna `sqlite:///overclock_study.db` storage — trial-by-trial writes mean the study survives a BSOD even if the live dashboard capture doesn't)
- **Visualization:** matplotlib + imageio, replaying the study database into animated history and Pareto-frontier GIFs after the fact
- **Platform:** Windows, AMD RDNA GPUs

---

## Setup & Build

### Prerequisites

- **OS:** Windows 10/11 x64
- **GPU:** Supported AMD RDNA discrete GPU with active AMD Adrenalin drivers
- **C++ Tools:** Visual Studio 2019/2022 with the **Desktop development with C++** workload
- **Python:** 3.10 or newer
- **Benchmark:** Cyberpunk 2077 installed, with the in-game benchmark accessible

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/<your-username>/Auto-Overclocker-AMD-GPU-.git
cd Auto-Overclocker-AMD-GPU-
pip install -r requirements.txt
```

### 2. Install the AMD ADLX SDK

The ADLX SDK is **not vendored in this repository** — `SDK/` is gitignored. Download it
yourself from [GPUOpen](https://gpuopen.com/adlx/) or the
[ADLX GitHub repo](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX).

Copy these three folders into the repository root, **preserving the exact directory layout**:

```
Auto-Overclocker-AMD-GPU-/
└── SDK/
    ├── ADLXHelper/
    │   └── Windows/
    │       └── Cpp/
    │           ├── ADLXHelper.cpp
    │           └── ADLXHelper.h
    ├── Include/
    │   └── (all ADLX headers)
    └── Platform/
        └── Windows/
            └── WinAPIs.cpp
```

> **The layout is not negotiable.** `ADLXHelper.cpp` reaches for `WinAPIs.h` by relative
> path (`../../../Platform/Windows/`), and `hardware_set.cpp` includes headers relative to
> the repo root. Flattening or renaming any of these folders produces either
> "cannot open include file" or a wall of `LNK2019: unresolved external symbol` errors.

There is no `.lib` to link against — ADLX resolves `amdadlx64.dll` from the installed
Adrenalin driver at runtime.

### 3. Build the hardware interface (C++)

Open **x64 Native Tools Command Prompt for VS** (the x64 prompt specifically — the x86
one will build a 32-bit binary that can't load the 64-bit ADLX DLL), `cd` to the repo
root, and run:

```bat
cl.exe /Zi /EHsc /nologo /I. /Fehardware_set.exe ^
  hardware_set.cpp ^
  SDK\ADLXHelper\Windows\Cpp\ADLXHelper.cpp ^
  SDK\Platform\Windows\WinAPIs.cpp
```

All three source files must be on the command line. Compiling `hardware_set.cpp` alone
links against nothing and fails.

> **Note on naming:** the build produces `hardware_set.exe` from `hardware_set.cpp`.
> Earlier revisions and parts of the demo video refer to this binary as
> `AutoOverclocker.exe` — same program, older name.

**Building from VS Code:** the repo ships a `.vscode/tasks.json` with the above already
configured. Press `Ctrl+Shift+B`. Do *not* use the built-in
"C/C++: cl.exe build active file" task — it compiles only the open editor tab and will
fail to link.

**Verify the build:**

```bat
hardware_set.exe 2500 2600 0 0 0
```

It should print the GPU it selected. The program enumerates all GPUs and picks the first
**discrete** one that reports manual tuning support, so integrated graphics are skipped
automatically — no hardcoded index.

### 4. Configure

Copy the template and edit your copy:

```bat
copy config.example.json config.json
```

**The file must be named `config.json`** — that exact name is what the loop reads.
`config.json` is gitignored (it holds machine-specific absolute paths), while
`config.example.json` is tracked, so your local settings never end up in a commit.

#### `paths`

All four are absolute paths, written with forward slashes.

| Key | What it points at |
|---|---|
| `cpp_executable` | The binary you built in step 3 — `hardware_set.exe` in the repo root |
| `cp2077_exe` | The Cyberpunk 2077 executable |
| `cp2077_dir` | The folder containing that executable (used as the working directory on launch) |
| `bench_dir` | Where the game writes benchmark results — replace `<YOUR_USERNAME>` with your Windows account name. **Cleared before every trial**, so point it at the game's results folder and nothing else. |

Steam paths differ if you installed to a secondary drive or use GOG/Epic; check the actual
install location rather than pasting the template values.

#### `tuning`

| Key | Default | What it does |
|---|---|---|
| `crash_penalty` | `-10000` | Score assigned to a configuration that killed the system. Large enough to dominate any real FPS result, so the sampler treats that region as strictly off-limits. |
| `voltage_penalty_weight` | `0.05` | How much undervolt is worth relative to FPS in the composite score. Raise it to bias the search toward deeper undervolts, lower it to chase raw framerate. |

#### `params`

The search bounds. Every configuration Optuna proposes falls inside these — they are the
only thing standing between the optimizer and your hardware's limits.

| Key | Units | Notes |
|---|---|---|
| `voltage_offset_min` / `_max` | mV | Undervolt offset, negative = less voltage. On RDNA 3/4 this is an offset from the driver's stock curve, not an absolute rail voltage. |
| `power_percentage_offset_min` / `_max` | % | Power limit offset relative to stock. Adrenalin typically caps this around ±15%. |
| `memory_freq_min` / `_max` | MHz | VRAM clock as ADLX reports it. |
| `gpu_freq_min` / `_max` | MHz | Maximum core clock target. |

Memory timing is also part of the search, but has no key here — it's binary (default vs.
fast), so the sampler treats it as a categorical choice rather than a range. Both values
are always explored.

**Start narrow and widen.** The shipped values are deliberately conservative — a range
that's already stable end to end wastes trials, but a range whose ceiling is far past
stable spends most of the run collecting BSODs. Find your rough manual limit in Adrenalin
first, then set the max slightly beyond it and let the sampler explore underneath.

Note that the tuned result in the [Results](#results) table used a -75 mV offset, which is
outside the template's `voltage_offset` range. Widen that bound to reproduce it.

### 5. Run the controller

```bash
python brain.py
```

Must be run **as Administrator** — the ADLX tuning calls require elevation to write to
the driver. The loop suggests a configuration, applies it, runs a benchmark pass, records
the result, and repeats.

Results stream to `overclock_study.db` (SQLite) after every trial, so the study survives
a hard crash. To resume an interrupted run, launch `brain.py` again — it reattaches to
the existing study rather than starting over.

### Resuming after a crash

Before each configuration is applied, it's journaled to `state.json` as `pending`. On the
next launch, `brain.py` reads any leftover `pending` entry, concludes the trial killed the
system, and reports it to Optuna with a heavy penalty so the sampler avoids that region.

Restarting after a BSOD is currently **manual** — rerun step 5 once Windows is back up.
Automatic relaunch on boot (via a startup batch script or Task Scheduler entry) is planned
but not yet implemented.

---

## Building the charts

The two GIFs at the top of this README are generated from `overclock_study.db` after a
run — nothing needs to be captured live. Because Optuna writes to SQLite trial-by-trial,
the study survives a BSOD even when a live screen capture doesn't, so these can be
rebuilt from a run that crashed halfway through.

```bash
python tools/rebuild_history_gif.py    # -> optimization_history.gif
python tools/pareto_history_gif.py     # -> pareto_front.gif
```

Both read the study directly and animate it one trial at a time. Frames are written to
`history_frames/` and `pareto_frames/` respectively (cleared on each run), then assembled
into a GIF. Edit the `CONFIG` block at the top of either file to change the study name,
frame rate, or output path.

| Script | Shows |
|---|---|
| `rebuild_history_gif.py` | Per-trial avg/min FPS, a step-line tracking the best trial so far by composite score, crash markers, and a live "current best configuration" card |
| `pareto_history_gif.py` | The score vs. \|voltage offset\| frontier filling in trial by trial, with dominated trials fading to gray |

Both scripts skip trials that are neither a recorded crash nor a completed benchmark, so
leftover `RUNNING` rows from an interrupted session don't appear as gaps.

**Note:** the best trial is ranked by composite score, not raw average FPS. A trial can
plot a higher FPS dot than the best-so-far line and still not be the leader — that's the
voltage penalty doing its job, not a bug.

These scripts need `matplotlib` and `imageio`, which are in `requirements.txt` but aren't
required to run the tuning loop itself.

---

⚠️ **This tool intentionally pushes GPU hardware past its stable operating envelope.** System freezes and BSODs are an expected, designed-for part of the search process, not a bug.

To make that safe(r) and self-recovering, the project includes:

- **Pending-state journaling** — before any parameter set is applied, it's written to `state.json` as `pending`. Nothing is marked successful until a benchmark pass actually completes.
- **Crash-aware resume** — a watchdog routine reads the `pending` entry left behind by a crashed trial on the next launch, so no trial is silently lost. (Relaunching after a BSOD is manual for now; see above.)
- **Automatic penalty scoring** — that fatal configuration is fed back into the Optuna study with a severe penalty (-10000), steering the optimizer away from that region of the search space instead of retrying it blindly.
- **Clean benchmark reads** — `bench_dir` is wiped before every trial, so a run that crashes before the game writes results can't be scored against the previous trial's leftover numbers.
- **Driver-safe teardown** — the ADLX wrapper includes deallocation workarounds to avoid memory access violations in the AMD driver DLL on thread termination, reducing crash frequency from the tooling itself (as opposed to the hardware limits being tested).
- **Sandbox limits** — you can manually set the sample space by editing `config.json`.

**Use at your own risk.** Overclocking/undervolting can cause system instability, data loss from unexpected reboots, and — in rare cases — hardware damage. Not affiliated with or endorsed by AMD, software is provided as is, test on hardware you're WILLING TO RISK. No warranty of any kind, express or implied.

---
