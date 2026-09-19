//
//-------------------------------------------------------------------------------------------------
// apply_tuning.cpp — applies one Optuna trial's GPU settings via ADLX.
//
// Error-handling design:
//   * Every failure maps to ONE exit code that tells brain.py what to do next
//     (fix the caller, prune the trial, penalize the trial, or stop the study).
//   * Two phases: validate everything first (nothing written), then apply.
//     An out-of-range value can never leave the GPU half-configured.
//   * Any failure during the apply phase rolls back to factory defaults; if the
//     rollback itself fails, that gets its own "stop everything" code.
//   * ADLX Terminate() happens exactly once, via RAII, after all ADLX smart
//     pointers have been released.
//-------------------------------------------------------------------------------------------------

#include "SDK/ADLXHelper/Windows/Cpp/ADLXHelper.h"
#include "SDK/Include/IGPUManualGFXTuning.h"
#include "SDK/Include/IGPUManualPowerTuning.h"
#include "SDK/Include/IGPUManualVRAMTuning.h"
#include "SDK/Include/IGPUTuning.h"
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

using namespace adlx;

static ADLXHelper g_ADLXHelp;

// =======================================================
// Exit-code contract with brain.py
// =======================================================
enum class Exit : int
{
    Ok             = 0,

    BadArgs        = 10, // caller bug (wrong argc / non-integer) -> fix brain.py, stop study

    AdlxInit       = 20, // environment problem  -> stop study
    NoGpu          = 21, //                      -> stop study
    Unsupported    = 22, // GPU lacks a feature  -> stop study, fix search space

    OutOfRange     = 30, // value outside driver range, NOTHING applied -> prune trial

    Rejected       = 40, // driver refused a value, rolled back to factory -> fail/penalize trial
    ResetNeeded    = 41, // driver asked for reset, rolled back to factory -> fail/penalize trial

    RollbackFailed = 50, // GPU state unknown -> stop study immediately
    Internal       = 60, // unexpected C++ exception -> stop study
};

struct TuneError
{
    Exit code;
    std::string msg;
};

[[noreturn]] static void fail(Exit code, const std::string& msg)
{
    throw TuneError{ code, msg };
}

// Turns an ADLX_RESULT into a TuneError. Use for reads/queries (never for applying values).
static void require(ADLX_RESULT r, Exit code, const std::string& what)
{
    if (ADLX_FAILED(r))
    {
        std::ostringstream m;
        m << what << " failed (ADLX_RESULT " << r << ")";
        fail(code, m.str());
    }
}

// =======================================================
// Argument parsing (std::stoi throws on bad input and would crash the process)
// =======================================================
struct Targets
{
    int coreMHz;
    int memMHz;
    int fastTiming;  // 0 = default, 1 = fast
    int powerLimit;
    int voltageMv;
};

static int parseInt(const char* s, const char* name)
{
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX)
        fail(Exit::BadArgs, std::string("Invalid integer for ") + name + ": '" + s + "'");
    return static_cast<int>(v);
}

static Targets parseArgs(int argc, char* argv[])
{
    if (argc != 6)
    {
        std::ostringstream m;
        m << "Expected 5 arguments, got " << (argc - 1)
          << ". Usage: " << argv[0]
          << " <max_gpu_freq> <mem_freq> <mem_timing(0/1)> <power_limit> <voltage>";
        fail(Exit::BadArgs, m.str());
    }

    Targets t{};
    t.coreMHz    = parseInt(argv[1], "max_gpu_freq");
    t.memMHz     = parseInt(argv[2], "mem_freq");
    t.fastTiming = parseInt(argv[3], "mem_timing");
    t.powerLimit = parseInt(argv[4], "power_limit");
    t.voltageMv  = parseInt(argv[5], "voltage");

    if (t.fastTiming != 0 && t.fastTiming != 1)
        fail(Exit::BadArgs, "mem_timing must be 0 or 1");

    return t;
}

// =======================================================
// Validation helpers
// =======================================================
static void checkRange(const char* name, int value, const ADLX_IntRange& r)
{
    if (value < r.minValue || value > r.maxValue)
    {
        std::ostringstream m;
        m << name << " = " << value << " is outside the driver range ["
          << r.minValue << ", " << r.maxValue << "]";
        fail(Exit::OutOfRange, m.str());
    }
    if (r.step > 1 && (value - r.minValue) % r.step != 0)
    {
        std::ostringstream m;
        m << name << " = " << value << " is not aligned to step " << r.step
          << " (starting at " << r.minValue << ")";
        fail(Exit::OutOfRange, m.str());
    }
}

// =======================================================
// GPU selection
// =======================================================
static IADLXGPUPtr selectGpu(IADLXSystem* sys, IADLXGPUTuningServicesPtr& tuning)
{
    IADLXGPUListPtr gpus;
    require(sys->GetGPUs(&gpus), Exit::NoGpu, "Enumerating GPUs");
    if (gpus->Empty())
        fail(Exit::NoGpu, "ADLX reported no GPUs");

    IADLXGPUPtr discrete;
    IADLXGPUPtr fallback;

    for (adlx_uint i = gpus->Begin(); i != gpus->End(); ++i)
    {
        IADLXGPUPtr gpu;
        if (ADLX_FAILED(gpus->At(i, &gpu)) || gpu == nullptr)
            continue;

        // Must actually be tunable: rules out the iGPU and any dGPU
        // the driver won't expose manual tuning for.
        adlx_bool tunable = false;
        if (ADLX_FAILED(tuning->IsSupportedManualGFXTuning(gpu, &tunable)) || !tunable)
            continue;

        if (!fallback)
            fallback = gpu;

        ADLX_GPU_TYPE type = GPUTYPE_UNDEFINED;
        if (ADLX_SUCCEEDED(gpu->Type(&type)) && type == GPUTYPE_DISCRETE)
        {
            discrete = gpu;
            break;
        }
    }

    // Some drivers report GPUTYPE_UNDEFINED; fall back to the first tunable GPU.
    IADLXGPUPtr chosen = discrete ? discrete : fallback;
    if (!chosen)
        fail(Exit::NoGpu, "No GPU with manual graphics tuning support found");

    const char* name = nullptr;
    if (ADLX_SUCCEEDED(chosen->Name(&name)) && name)
        std::cout << "Selected GPU: " << name << std::endl;

    return chosen;
}

// =======================================================
// Main work. Every ADLX smart pointer lives inside this function,
// so all of them are released before ADLX is terminated.
// =======================================================
static void run(const Targets& t)
{
    IADLXSystem* sys = g_ADLXHelp.GetSystemServices();
    if (!sys)
        fail(Exit::AdlxInit, "GetSystemServices returned null");

    IADLXGPUTuningServicesPtr tuning;
    require(sys->GetGPUTuningServices(&tuning), Exit::AdlxInit, "Getting GPU tuning services");

    IADLXGPUPtr gpu = selectGpu(sys, tuning);

    // ---------------------------------------------------
    // Phase 1: acquire every interface, read every range,
    //          validate every value. Nothing is written yet.
    // ---------------------------------------------------

    // Graphics (core clock + voltage) — RDNA-style interface required.
    IADLXInterfacePtr gfxIfc;
    require(tuning->GetManualGFXTuning(gpu, &gfxIfc), Exit::Unsupported, "Getting manual GFX tuning");
    IADLXManualGraphicsTuning2Ptr gfx(gfxIfc);
    if (!gfx)
        fail(Exit::Unsupported, "GPU does not expose IADLXManualGraphicsTuning2 (pre-RDNA GPU?)");

    // VRAM (frequency + timing)
    adlx_bool supported = false;
    if (ADLX_FAILED(tuning->IsSupportedManualVRAMTuning(gpu, &supported)) || !supported)
        fail(Exit::Unsupported, "Manual VRAM tuning is not supported on this GPU");
    IADLXInterfacePtr vramIfc;
    require(tuning->GetManualVRAMTuning(gpu, &vramIfc), Exit::Unsupported, "Getting manual VRAM tuning");
    IADLXManualVRAMTuning2Ptr vram(vramIfc);
    if (!vram)
        fail(Exit::Unsupported, "GPU does not expose IADLXManualVRAMTuning2");

    adlx_bool timingSupported = false;
    if (ADLX_FAILED(vram->IsSupportedMemoryTiming(&timingSupported)))
        timingSupported = false;
    if (t.fastTiming == 1 && !timingSupported)
        fail(Exit::Unsupported, "Fast memory timing requested but not supported; remove it from the search space");

    // Power
    supported = false;
    if (ADLX_FAILED(tuning->IsSupportedManualPowerTuning(gpu, &supported)) || !supported)
        fail(Exit::Unsupported, "Manual power tuning is not supported on this GPU");
    IADLXInterfacePtr powerIfc;
    require(tuning->GetManualPowerTuning(gpu, &powerIfc), Exit::Unsupported, "Getting manual power tuning");
    IADLXManualPowerTuningPtr power(powerIfc);
    if (!power)
        fail(Exit::Unsupported, "GPU does not expose IADLXManualPowerTuning");

    // Ranges
    ADLX_IntRange coreR{}, voltR{}, memR{}, powR{};
    require(gfx->GetGPUMaxFrequencyRange(&coreR), Exit::Unsupported, "Reading core clock range");
    require(gfx->GetGPUVoltageRange(&voltR),      Exit::Unsupported, "Reading voltage range");
    require(vram->GetMaxVRAMFrequencyRange(&memR), Exit::Unsupported, "Reading VRAM clock range");
    require(power->GetPowerLimitRange(&powR),     Exit::Unsupported, "Reading power limit range");

    checkRange("max_gpu_freq", t.coreMHz,    coreR);
    checkRange("voltage",      t.voltageMv,  voltR);
    checkRange("mem_freq",     t.memMHz,     memR);
    checkRange("power_limit",  t.powerLimit, powR);

    // ---------------------------------------------------
    // Phase 2: apply. Any failure rolls back to factory
    //          so the next trial starts from a known state.
    // ---------------------------------------------------
    auto apply = [&](ADLX_RESULT r, const char* what)
    {
        if (ADLX_SUCCEEDED(r))
            return;

        std::ostringstream m;
        m << "Driver rejected " << what << " (ADLX_RESULT " << r << ")";
        const Exit code = (r == ADLX_RESET_NEEDED) ? Exit::ResetNeeded : Exit::Rejected;

        ADLX_RESULT rr = tuning->ResetToFactory(gpu);
        if (ADLX_FAILED(rr))
        {
            m << "; ResetToFactory ALSO failed (ADLX_RESULT " << rr << "). GPU state is unknown.";
            fail(Exit::RollbackFailed, m.str());
        }
        m << "; restored factory defaults";
        fail(code, m.str());
    };

    apply(gfx->SetGPUMaxFrequency(t.coreMHz),  "max GPU frequency");
    apply(gfx->SetGPUVoltage(t.voltageMv),     "GPU voltage");
    apply(vram->SetMaxVRAMFrequency(t.memMHz), "VRAM frequency");
    if (timingSupported)
    {
        ADLX_MEMORYTIMING_DESCRIPTION desc = t.fastTiming ? MEMORYTIMING_FAST_TIMING : MEMORYTIMING_DEFAULT;
        apply(vram->SetMemoryTimingDescription(desc), "memory timing");
    }
    apply(power->SetPowerLimit(t.powerLimit), "power limit");
}

// =======================================================
// the only place that turns errors into exit codes.
// =======================================================
int main(int argc, char* argv[])
{
    try
    {
        const Targets t = parseArgs(argc, argv);

        std::cout << "Target -> Core: " << t.coreMHz << "MHz, VRAM: " << t.memMHz
                  << "MHz, Timing: " << (t.fastTiming ? "Fast" : "Default")
                  << ", Power: " << t.powerLimit << ", Voltage: " << t.voltageMv << "mV" << std::endl;

        if (ADLX_FAILED(g_ADLXHelp.Initialize()))
            fail(Exit::AdlxInit, "ADLX failed to initialize (AMD driver missing or too old?)");

        // Terminates ADLX exactly once, on success or on any throw,
        // after run()'s ADLX pointers have already been released.
        struct AdlxSession { ~AdlxSession() { g_ADLXHelp.Terminate(); } } session;

        run(t);

        std::cout << "[SUCCESS] All parameters applied." << std::endl;
        return static_cast<int>(Exit::Ok);
    }
    catch (const TuneError& e)
    {
        std::cerr << "[FAIL " << static_cast<int>(e.code) << "] " << e.msg << std::endl;
        return static_cast<int>(e.code);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL " << static_cast<int>(Exit::Internal) << "] Unexpected exception: " << e.what() << std::endl;
        return static_cast<int>(Exit::Internal);
    }
}
