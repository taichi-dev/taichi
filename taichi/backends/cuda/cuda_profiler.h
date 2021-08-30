#pragma once

#include "taichi/program/kernel_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "cuda_profiler_functions.h"

#include <string>
#include <stdint.h>

TLANG_NAMESPACE_BEGIN

typedef enum CUDA_KERNEL_PROFILER {
  CUDA_KERNEL_PROFILER_UNDEF  = 0,
  CUDA_KERNEL_PROFILER_EVENT  = 1,
  CUDA_KERNEL_PROFILER_CUPTI  = 2
} CUDAKernalProfiler;


struct profilerConfig
{
  bool enable = false;
  CUDAKernalProfiler profiler_type = CUDA_KERNEL_PROFILER_UNDEF;
  KernelProfilerMode profiling_mode = KernelProfilerMode::disable;
  uint32_t num_ranges = 1024*16;
  std::vector<std::string> metric_names;
#if defined(TI_WITH_TOOLKIT_CUDA)
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

struct profilerRawData
{
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
  
  bool set_profiling_mode(KernelProfilerMode profiling_mode);
  CUDAKernalProfiler get_profiler_type(){return profiler_config_.profiler_type;}
  KernelProfilerMode get_profiling_mode(){return profiler_config_.profiling_mode;}

  bool init_cupti();
  bool deinit_cupti();
  bool begin_profiling();
  bool end_profiling();

  void record_launched_kernel(std::string name){
    CUDAKernelTracedRecord record;
    record.kernel_name = name;
    traced_records_.push_back(record);
  }

  bool traceMetricValues();
  bool statisticsOnRecords(std::vector<KernelProfileRecord> &records,double &total_time_ms);
  bool resetCUPTIprofiler();
  bool printTracedRecords();
  void clearTracedRecords(){ 
    traced_records_.clear(); 
  }

private:
  profilerConfig profiler_config_;
  profilerRawData profiler_data_;
  std::vector<CUDAKernelTracedRecord> traced_records_;

  // std::mutex lock;
};

TLANG_NAMESPACE_END