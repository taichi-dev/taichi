#pragma once

// #include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/common/core.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

struct CuptiConfig {
  bool enable = false;
#if defined(TI_WITH_CUDA_TOOLKIT)
  uint32_t num_ranges = 16384;  // max number of kernels traced by CUPTI
  std::vector<std::string> metric_list;
  // CUpti_ProfilerRange profiler_range = CUPTI_AutoRange;
  // CUpti_ProfilerReplayMode profiler_replay_mode = CUPTI_KernelReplay;
#endif
};

struct CuptiImage {
  std::string chip_name;
  std::vector<uint8_t> config_image;
  std::vector<uint8_t> counter_availability_image;
  std::vector<uint8_t> counter_data_scratch_buffer;
  std::vector<uint8_t> counter_data_image_prefix;
  std::vector<uint8_t> counter_data_image;
};

class CuptiToolkit {
 public:
  CuptiToolkit();
  ~CuptiToolkit();

  void set_enable();

  bool init_cupti();
  bool deinit_cupti();
  bool begin_profiling();
  bool end_profiling();
  bool calculate_metric_values();

 private:
  CuptiConfig cupti_config_;
  CuptiImage cupti_image_;
};

TLANG_NAMESPACE_END
