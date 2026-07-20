# Automatic GPU Overclocker (AMD GPU)
An automation framework that uses Bayesian optimization (Optuna) and AMD's native C++ API to find stable GPU overclocking and undervolting settings.


## Demo


<img width="1045" height="880" alt="optimization_history" src="https://github.com/user-attachments/assets/e2c72e26-9880-42f3-a79b-0050c66129d9" />

50 trials of the Optuna sampler searching the clock/voltage space (note that the 0th trial is a default configuration/baseline run). The gif shows every trial's avg/min FPS (faint dots), a red step-line tracking the best-so-far trial by composite score, and red X's marking trials that crashed the system before producing a benchmark. The overclock

[![Watch the video](https://img.youtube.com/vi/YyaOEYRJmVg/maxresdefault.jpg)](https://youtu.be/YyaOEYRJmVg)
A video demo of the program recovering from a crash, setting gpu parameters, running a benchmark, logging results, and deciding the next test gpu parameters. 
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

The Pareto frontier evolves as trials accumulate — new configurations join the front and displace previously "best" trade-offs, while the overall best trial (gold star) only ever holds steady or improves, climbing from a score*[remember to finish the apsotrophe] of 65.72 to 70.34 by trial 51.

## How It Works

The system separates concerns into a high-level ML controller and a low-level hardware interface, connected by a persistent, crash-resilient execution loop:

1. **Suggest** — `brain.py` (Optuna, multivariate TPE sampler) proposes a candidate parameter set (core clock, memory clock, memory timing, voltage offset, power limits).
2. **Apply** — `AutoOverclocker.exe` (compiled from `hardware_set.cpp`, AMD ADLX SDK) injects the parameters directly into the GPU driver.
3. **Benchmark** — a Python helper launches a Cyberpunk 2077 benchmark pass and captures FPS metrics.
4. **Log & Iterate** — results (or a recorded crash) are written back to the Optuna study; the model updates and proposes the next candidate.
<svg width="100%" viewBox="0 0 680 610" role="img" style="" xmlns="http://www.w3.org/2000/svg">
<title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Auto overclocker architecture</title>
<desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">Flowchart of the GPU auto-overclocker loop: config.json feeds Optuna's suggest step, which flows to Apply, Benchmark, and Log &amp; score, looping back to Suggest. A watchdog branches off Apply to catch crashes and feed a penalty back into Suggest.</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker>
</defs>

<line x1="280" y1="96" x2="280" y2="156" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="280" y1="212" x2="280" y2="272" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="280" y1="328" x2="280" y2="388" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="280" y1="444" x2="280" y2="504" marker-end="url(#arrow)" style="fill:none;stroke:rgb(137, 135, 129);color:rgb(11, 11, 11);stroke-width:1.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<line x1="390" y1="300" x2="440" y2="300" stroke="#D85A30" stroke-width="1" marker-end="url(#arrow)" style="fill:rgb(0, 0, 0);stroke:rgb(216, 90, 48);color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="415" y="290" text-anchor="middle" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:auto">crash</text>

<path d="M540,272 L540,230 L390,230 L390,184" fill="none" stroke="#E24B4A" stroke-width="1" stroke-dasharray="4 4" marker-end="url(#arrow)" style="fill:none;stroke:rgb(226, 75, 74);color:rgb(11, 11, 11);stroke-width:1px;stroke-dasharray:4px, 4px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<path d="M170,532 C70,532 70,184 170,184" fill="none" stroke="#7F77DD" stroke-width="1" marker-end="url(#arrow)" style="fill:none;stroke:rgb(127, 119, 221);color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="105" y="360" text-anchor="start" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:start;dominant-baseline:auto">↻ next trial</text>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="170" y="40" width="220" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="280" y="60" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">config.json</text>
<text x="280" y="80" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Search space &amp; penalties</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="170" y="156" width="220" height="56" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="280" y="176" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Suggest</text>
<text x="280" y="196" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">TPE picks next params</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="170" y="272" width="220" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="280" y="292" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Apply</text>
<text x="280" y="312" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Writes pending, sets GPU</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="440" y="272" width="200" height="56" rx="8" stroke-width="0.5" style="fill:rgb(252, 235, 235);stroke:rgb(163, 45, 45);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="540" y="292" text-anchor="middle" dominant-baseline="central" style="fill:rgb(121, 31, 31);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Watchdog</text>
<text x="540" y="312" text-anchor="middle" dominant-baseline="central" style="fill:rgb(163, 45, 45);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Penalizes crashed trial</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="170" y="388" width="220" height="56" rx="8" stroke-width="0.5" style="fill:rgb(241, 239, 232);stroke:rgb(95, 94, 90);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="280" y="408" text-anchor="middle" dominant-baseline="central" style="fill:rgb(68, 68, 65);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Benchmark</text>
<text x="280" y="428" text-anchor="middle" dominant-baseline="central" style="fill:rgb(95, 94, 90);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Runs &amp; verifies benchmark</text>
</g>

<g style="fill:rgb(0, 0, 0);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">
<rect x="170" y="504" width="220" height="56" rx="8" stroke-width="0.5" style="fill:rgb(238, 237, 254);stroke:rgb(83, 74, 183);color:rgb(11, 11, 11);stroke-width:0.5px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="280" y="524" text-anchor="middle" dominant-baseline="central" style="fill:rgb(60, 52, 137);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:14px;font-weight:500;text-anchor:middle;dominant-baseline:central">Log &amp; score</text>
<text x="280" y="544" text-anchor="middle" dominant-baseline="central" style="fill:rgb(83, 74, 183);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:central">Writes score, logs result</text>
</g>

<text x="280" y="590" text-anchor="middle" style="fill:rgb(82, 81, 78);stroke:none;color:rgb(11, 11, 11);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, sans-serif;font-size:12px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Purple = optimizer step · Red = crash recovery</text>
</svg>
<img width="300" height="269" alt="gpu_overclocker_architecture" src="https://github.com/user-attachments/assets/d35b4bf9-2267-402c-be49-0f554132b870" />

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
