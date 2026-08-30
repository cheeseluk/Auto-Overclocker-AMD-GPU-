//
//-------------------------------------------------------------------------------------------------

#include "SDK/ADLXHelper/Windows/Cpp/ADLXHelper.h"
#include "SDK/Include/IGPUManualGFXTuning.h"
#include "SDK/Include/IGPUManualPowerTuning.h"
#include "SDK/Include/IGPUManualVRAMTuning.h"
#include "SDK/Include/IGPUTuning.h"
#include <iostream>
#include <string>

// Use ADLX namespace
using namespace adlx;

// ADLXHelper instance
static ADLXHelper g_ADLXHelp;

int main(int argc, char* argv[])
{
    // 1. Check if Python provided all 5 arguments
    if (argc < 6)
    {
        std::cerr << "Usage: " << argv[0] << " <max_gpu_freq> <mem_freq> <mem_timing(0/1)> <power_limit> <voltage>" << std::endl;
        return 1; // Return failure to Optuna
    }

    // Parse the inputs from brain.py
    int target_max_freq = std::stoi(argv[1]);
    int target_mem_freq = std::stoi(argv[2]);
    int target_mem_timing = std::stoi(argv[3]); // 0 = Default, 1 = Fast Timing
    int target_power_limit = std::stoi(argv[4]);
    int target_voltage = std::stoi(argv[5]);

    std::cout << "Target -> Core: " << target_max_freq << "MHz, VRAM: " << target_mem_freq
        << "MHz, Timing: " << (target_mem_timing == 1 ? "Fast" : "Default")
        << ", Power: " << target_power_limit << ", Voltage: " << target_voltage << "mV" << std::endl;

    // 2. Initialize ADLX
    ADLX_RESULT res = g_ADLXHelp.Initialize();
    if (ADLX_FAILED(res)) return 1;

    // 3. Get the GPU Tuning Service
    IADLXGPUTuningServicesPtr gpuTuningService;
    res = g_ADLXHelp.GetSystemServices()->GetGPUTuningServices(&gpuTuningService);
    if (ADLX_FAILED(res)) { g_ADLXHelp.Terminate(); return 1; }

    // 4. Find the discrete, tunable GPU
    IADLXGPUListPtr gpus;
    res = g_ADLXHelp.GetSystemServices()->GetGPUs(&gpus);
    if (ADLX_FAILED(res) || gpus->Empty()) { g_ADLXHelp.Terminate(); return 1; }

    IADLXGPUPtr oneGPU;
    IADLXGPUPtr fallbackGPU;

    for (adlx_uint i = gpus->Begin(); i != gpus->End(); ++i)
    {
        IADLXGPUPtr gpu;
        if (ADLX_FAILED(gpus->At(i, &gpu)) || gpu == nullptr)
            continue;

        // Must actually be tunable — rules out the iGPU and any
        // dGPU the driver won't expose manual tuning for.
        adlx_bool tunable = false;
        if (ADLX_FAILED(gpuTuningService->IsSupportedManualGFXTuning(gpu, &tunable)) || !tunable)
            continue;

        if (!fallbackGPU)
            fallbackGPU = gpu;

        ADLX_GPU_TYPE type = GPUTYPE_UNDEFINED;
        if (ADLX_SUCCEEDED(gpu->Type(&type)) && type == GPUTYPE_DISCRETE)
        {
            oneGPU = gpu;
            break;
        }
    }

    // Some drivers report GPUTYPE_UNDEFINED; fall back to the first tunable GPU.
    if (!oneGPU)
        oneGPU = fallbackGPU;

    if (oneGPU == nullptr)
    {
        std::cerr << "[ERROR] No tunable discrete GPU found." << std::endl;
        g_ADLXHelp.Terminate();
        return 1;
    }

    const char* gpuName = nullptr;
    if (ADLX_SUCCEEDED(oneGPU->Name(&gpuName)) && gpuName)
        std::cout << "Selected GPU: " << gpuName << std::endl;

        auto applyAndCheck = [&](ADLX_RESULT r, const char* settingName) -> bool {
        if (r == ADLX_RESET_NEEDED) {
            std::cerr << "[CRASH] Driver rejected " << settingName << "! Resetting to safe defaults..." << std::endl;
            gpuTuningService->ResetToFactory(oneGPU);
            return true;
        }
        if (ADLX_FAILED(r)) {
            std::cerr << "[ERROR] Failed to apply " << settingName << " (Code: " << r << ")" << std::endl;
            return true;
        }
        return false;
    };

    // =======================================================
    // A. Apply Graphics Tuning (Core Clock & Voltage)
    // =======================================================
    IADLXInterfacePtr gfxTuningIfc;
    if (ADLX_SUCCEEDED(gpuTuningService->GetManualGFXTuning(oneGPU, &gfxTuningIfc)))
    {// Using Post-Navi ASIC Interface (RDNA series)
        IADLXManualGraphicsTuning2Ptr gfxTuning(gfxTuningIfc);
        if (gfxTuning)
        {
            res = gfxTuning->SetGPUMaxFrequency(target_max_freq);
            if (applyAndCheck(res, "Max GPU Frequency")) { g_ADLXHelp.Terminate(); return 1; }

            res = gfxTuning->SetGPUVoltage(target_voltage);
            if (applyAndCheck(res, "GPU Voltage")) { g_ADLXHelp.Terminate(); return 1; }
        }
    }

    // =======================================================
    // B. Apply VRAM Tuning (Memory Frequency & Timings)
    // =======================================================
    IADLXInterfacePtr vramTuningIfc;
    if (ADLX_SUCCEEDED(gpuTuningService->GetManualVRAMTuning(oneGPU, &vramTuningIfc)))
    {
        IADLXManualVRAMTuning2Ptr vramTuning(vramTuningIfc);
        if (vramTuning)
        {
            res = vramTuning->SetMaxVRAMFrequency(target_mem_freq);
            if (applyAndCheck(res, "VRAM Frequency")) { g_ADLXHelp.Terminate(); return 1; }

            ADLX_MEMORYTIMING_DESCRIPTION desc = (target_mem_timing == 1) ? MEMORYTIMING_FAST_TIMING : MEMORYTIMING_DEFAULT;
            res = vramTuning->SetMemoryTimingDescription(desc);
            if (applyAndCheck(res, "Memory Timing")) { g_ADLXHelp.Terminate(); return 1; }
        }
    }

    // =======================================================
    // C. Apply Power Tuning (Power Limit)
    // =======================================================
    IADLXInterfacePtr powerTuningIfc;
    if (ADLX_SUCCEEDED(gpuTuningService->GetManualPowerTuning(oneGPU, &powerTuningIfc)))
    {
        IADLXManualPowerTuningPtr powerTuning(powerTuningIfc);
        if (powerTuning)
        {
            res = powerTuning->SetPowerLimit(target_power_limit);
            if (applyAndCheck(res, "Power Limit")) { g_ADLXHelp.Terminate(); return 1; }
        }
    }

    // 5. Clean Exit
    std::cout << "[SUCCESS] All parameters applied gracefully." << std::endl;
    g_ADLXHelp.Terminate();

    return 0; // Return 0 back to brain.py so Optuna knows it's time to run the benchmark
}