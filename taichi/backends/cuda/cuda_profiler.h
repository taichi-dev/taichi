#pragma once

#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "cuda_profiler_functions.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

typedef enum CUDAKernalProfiler {
  CUDA_KERNEL_PROFILER_UNDEF = 0,
  CUDA_KERNEL_PROFILER_EVENT = 1,
  CUDA_KERNEL_PROFILER_CUPTI = 2
};

struct ProfilerConfig {
  CUDAKernalProfiler profiler_type = CUDA_KERNEL_PROFILER_UNDEF;
  KernelProfilerMode profiling_mode = KernelProfilerMode::disable;
#if defined(TI_WITH_TOOLKIT_CUDA)
  uint32_t num_ranges = 16384;  // max number of kernels traced by CUPTI
  std::vector<std::string> metric_list;  // metric name list
  CUpti_ProfilerRange profiler_range = CUPTI_AutoRange;
  CUpti_ProfilerReplayMode profiler_replay_mode = CUPTI_KernelReplay;
#endif
};

struct CUDAKernelTracedRecord {
  std::string kernel_name;
  float kernel_elapsed_time_in_ms{0.0};
  float kernel_gloabl_load_byets{0.0};
  float kernel_gloabl_store_byets{0.0};
  float utilization_ratio_sm{0.0};
  float utilization_ratio_mem{0.0};
};

struct ProfilerRawData {
  std::string chip_name;
  std::vector<uint8_t> config_image;
  std::vector<uint8_t> counter_availability_image;
  std::vector<uint8_t> counter_data_scratch_buffer;
  std::vector<uint8_t> counter_data_image_prefix;
  std::vector<uint8_t> counter_data_image;
};

class CUDAProfiler {
 public:
  CUDAProfiler(KernelProfilerMode &mode);
  ~CUDAProfiler();

  bool is_cuda_profiler(KernelProfilerMode profiling_mode);
  bool set_profiler(KernelProfilerMode &profiling_mode);

  KernelProfilerMode get_profiling_mode();
  CUDAKernalProfiler get_profiler_type();

  bool init_cupti();
  bool deinit_cupti();
  bool begin_profiling();
  bool end_profiling();

  void record_launched_kernel(std::string name);
  bool calculate_metric_values();
  bool statistics_on_traced_records(std::vector<KernelProfileRecord> &records,
                                    double &total_time_ms);
  void clear_traced_records();

 private:
  ProfilerConfig profiler_config_;
  ProfilerRawData profiler_data_;
  std::vector<CUDAKernelTracedRecord> traced_records_;
};

TLANG_NAMESPACE_END
