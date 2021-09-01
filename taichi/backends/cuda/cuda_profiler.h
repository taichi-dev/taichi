#pragma once

#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "cuda_profiler_functions.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

typedef enum CUDA_KERNEL_PROFILER {
  CUDA_KERNEL_PROFILER_UNDEF = 0,
  CUDA_KERNEL_PROFILER_EVENT = 1,
  CUDA_KERNEL_PROFILER_CUPTI = 2
} CUDAKernalProfiler;

struct ProfilerConfig {
  CUDAKernalProfiler profiler_type = CUDA_KERNEL_PROFILER_UNDEF;
  KernelProfilerMode profiling_mode = KernelProfilerMode::disable;
#if defined(TI_WITH_TOOLKIT_CUDA)
  uint32_t num_ranges = 16384; //max number of kernels traced by CUPTI
  std::vector<std::string> metric_list; //metric name list
  CUpti_ProfilerRange profiler_range = CUPTI_AutoRange;
  CUpti_ProfilerReplayMode profiler_replay_mode = CUPTI_KernelReplay;
#endif
};

struct CUDAKernelTracedRecord {
  std::string kernel_name;
  float kernel_elapsed_time_in_ms;
  double kernel_gloabl_load_byets;
  double kernel_gloabl_store_byets;
  float utilization_ratio_sm;
  float utilization_ratio_mem;
};

struct ProfilerRawData {
  std::string chipName;
  std::vector<uint8_t> counterDataImagePrefix;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::vector<uint8_t> counterDataScratchBuffer;
  std::vector<uint8_t> counterAvailabilityImage;
};

class CUDAProfiler {
 public:
  CUDAProfiler();
  ~CUDAProfiler();

  static CUDAProfiler &get_instance();
  static CUDAProfiler &get_instance_without_context();

  bool is_cuda_profiler(KernelProfilerMode profiling_mode);
  bool set_profiler(KernelProfilerMode profiling_mode);

  KernelProfilerMode get_profiling_mode();
  CUDAKernalProfiler get_profiler_type();

  bool init_cupti();
  bool deinit_cupti();
  bool begin_profiling();
  bool end_profiling();

  void record_launched_kernel(std::string name);
  bool trace_metric_values();
  bool statistics_on_traced_records(std::vector<KernelProfileRecord> &records,
                           double &total_time_ms);
  void clear_traced_records();

 private:
  ProfilerConfig profiler_config_;
  ProfilerRawData profiler_data_;
  std::vector<CUDAKernelTracedRecord> traced_records_;
};

TLANG_NAMESPACE_END
