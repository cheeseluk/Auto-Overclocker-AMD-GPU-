"""
Runs the ADLX tuner executable and turns its exit code into a Python exception.

Contract (must match the Exit enum in apply_tuning.cpp):
    0        success
    10       bad arguments              -> FatalError   (bug in brain.py)
    20-22    ADLX / GPU / feature issue -> FatalError   (environment problem)
    30       value out of driver range  -> InvalidConfig (nothing applied, prune trial)
    40, 41   driver rejected / reset    -> UnstableConfig (rolled back, penalize trial)
    50       rollback failed            -> FatalError   (GPU state unknown)
    60       internal tuner error       -> FatalError
    other    tuner process crashed      -> FatalError
"""

import subprocess

TUNER_EXE = "hardware_set.exe"  # <- keep whatever path your previous version used
APPLY_TIMEOUT_S = 30


class TuningError(Exception):
    """Base class for everything this module raises."""


class InvalidConfig(TuningError):
    """Settings are outside what the driver accepts. Nothing was applied."""


class UnstableConfig(TuningError):
    """Driver rejected the settings or crashed on them. GPU was reset to factory."""


class FatalError(TuningError):
    """The harness or GPU is in a state where continuing the study is unsafe or pointless."""


_FATAL_CODES = {10, 20, 21, 22, 60}


def apply_gpu_settings(trial_number, params):
    """Apply one trial's settings. Returns None on success, raises a TuningError otherwise."""
    args = [
        TUNER_EXE,
        str(params["freq_gpu"]),
        str(params["memfreq"]),
        str(params["mem_timing"]),
        str(params["power_limit"]),
        str(params["voltage"]),
    ]

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=APPLY_TIMEOUT_S)
    except FileNotFoundError as e:
        raise FatalError(f"Tuner executable not found: {TUNER_EXE}") from e
    except subprocess.TimeoutExpired as e:
        raise FatalError(
            f"Tuner hung for more than {APPLY_TIMEOUT_S}s; the driver may be stuck. "
            "Reboot before running again."
        ) from e

    for line in result.stdout.splitlines():
        print(f"[Trial {trial_number}] {line}")

    code = result.returncode
    detail = result.stderr.strip() or f"exit code {code}"

    if code == 0:
        return
    if code == 30:
        raise InvalidConfig(detail)
    if code in (40, 41):
        raise UnstableConfig(detail)
    if code == 50:
        raise FatalError(
            f"{detail}\nReset the GPU in AMD Software (or reboot) before running again."
        )
    if code in _FATAL_CODES:
        raise FatalError(detail)

    # Anything else means the tuner process itself died (e.g. 0xC0000005 access violation).
    raise FatalError(
        f"Tuner exited with unexpected code {code} (0x{code & 0xFFFFFFFF:08X}); "
        f"it most likely crashed. {detail}"
    )