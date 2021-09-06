// Supported kernel profiler mode

PER_MODE(disable)           // common      : False
PER_MODE(enable)            // common      : True  [CUDA:cuEvent]
PER_MODE(cupti_onepass)     // CUDA        : CUPTI::AccurateMode
PER_MODE(cupti_detailed)    // CUDA        : CUPTI::DetailedMode
PER_MODE(cupti_customized)  // CUDA        : CUPTI::DetailedMode
