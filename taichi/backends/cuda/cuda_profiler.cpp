#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"

TLANG_NAMESPACE_BEGIN

CUDAProfiler::CUDAProfiler() {
  TI_TRACE("CUDAProfiler::CUDAProfiler() ");
}

CUDAProfiler::~CUDAProfiler() {
  if (profiler_config_.profiler_type == CUDA_KERNEL_PROFILER_CUPTI) {
    end_profiling();
    deinit_cupti();
  }
}

bool CUDAProfiler::is_cuda_profiler(KernelProfilerMode profiling_mode) {
  bool ret = profiling_mode == KernelProfilerMode::enable |
             profiling_mode == KernelProfilerMode::cuda_accurate |
             profiling_mode == KernelProfilerMode::cuda_detailed;
  return ret;
}

bool CUDAProfiler::set_profiler(KernelProfilerMode &profiling_mode) {
  if (!is_cuda_profiler(profiling_mode)) {
    return false;
  }

  CUDAKernalProfiler profiler_type =
      (profiling_mode == KernelProfilerMode::enable)
          ? CUDA_KERNEL_PROFILER_CUPTI
          : CUDA_KERNEL_PROFILER_EVENT;

  if (profiling_mode == KernelProfilerMode::enable) {
    profiler_config_.profiler_type = CUDA_KERNEL_PROFILER_EVENT;
    profiler_config_.profiling_mode = KernelProfilerMode::enable;
    TI_TRACE("CUDA_KERNEL_PROFILER_EVENT : enable");
    TI_TRACE("profiler_type : {}", profiler_config_.profiler_type);
    profiling_mode = profiler_config_.profiling_mode;
    return true;
  }

#if !defined(TI_WITH_TOOLKIT_CUDA)
  if (profiler_type == CUDA_KERNEL_PROFILER_CUPTI) {
    TI_WARN(
        "CUPTI toolkit is not compiled with taichi, fallback to cuEvent kernel "
        "profiler");
    TI_WARN(
        "to use CUPTI kernel profiler : "
        "TAICHI_CMAKE_ARGS=-DTI_WITH_TOOLKIT_CUDA=True python3 setup.py "
        "develop --user");

    profiler_config_.profiler_type = CUDA_KERNEL_PROFILER_EVENT;
    profiler_config_.profiling_mode = KernelProfilerMode::enable;

    TI_TRACE("CUDA_KERNEL_PROFILER_EVENT : enable");
    TI_TRACE("profiler_type : {}", profiler_config_.profiler_type);
    profiling_mode = profiler_config_.profiling_mode;
    return true;
  }
#else
// TODO::CUPTI_PROFILER
#endif
}

CUDAKernalProfiler CUDAProfiler::get_profiler_type() {
  return profiler_config_.profiler_type;
}

KernelProfilerMode CUDAProfiler::get_profiling_mode() {
  return profiler_config_.profiling_mode;
}

void CUDAProfiler::record_launched_kernel(std::string name) {
  CUDAKernelTracedRecord record;
  record.kernel_name = name;
  traced_records_.push_back(record);
}

bool CUDAProfiler::statistics_on_traced_records(
    std::vector<KernelProfileRecord> &records,
    double &total_time_ms) {
  size_t records_size = traced_records_.size();
  if (!records_size) {
    TI_WARN("traced_records_ is empty!");
    return false;
  }

  for (size_t resultIndex = 0; resultIndex < records_size; ++resultIndex) {
    auto it = std::find_if(
        records.begin(), records.end(), [&](KernelProfileRecord &r) {
          return r.name == traced_records_[resultIndex].kernel_name;
        });
    if (it == records.end()) {
      records.push_back(traced_records_[resultIndex].kernel_name);
      it = std::prev(records.end());
    }
    it->insert_sample(traced_records_[resultIndex].kernel_elapsed_time_in_ms);
    total_time_ms += traced_records_[resultIndex].kernel_elapsed_time_in_ms;

    if (profiler_config_.profiling_mode == KernelProfilerMode::cuda_accurate) {
      it->cuda_mem_access(
          traced_records_[resultIndex].kernel_gloabl_load_byets,
          traced_records_[resultIndex].kernel_gloabl_store_byets);
    }
    if (profiler_config_.profiling_mode == KernelProfilerMode::cuda_detailed) {
      it->cuda_mem_access(
          traced_records_[resultIndex].kernel_gloabl_load_byets,
          traced_records_[resultIndex].kernel_gloabl_store_byets);
      it->cuda_utilization_ratio(
          traced_records_[resultIndex].utilization_ratio_sm,
          traced_records_[resultIndex].utilization_ratio_mem);
    }
  }

  return true;
}

void CUDAProfiler::clear_traced_records() {
  traced_records_.clear();
}

CUDAProfiler &CUDAProfiler::get_instance_without_context() {
  static CUDAProfiler *instance = new CUDAProfiler();
  return *instance;
}

CUDAProfiler &CUDAProfiler::get_instance() {
  CUDAContext::get_instance();
  return get_instance_without_context();
}

#if defined(TI_WITH_TOOLKIT_CUDA)
// TODO::CUPTI_PROFILER
#else
bool CUDAProfiler::init_cupti(){};  // TODO TI_WARN
bool CUDAProfiler::begin_profiling(){};
bool CUDAProfiler::end_profiling(){};
bool CUDAProfiler::deinit_cupti(){};
bool CUDAProfiler::calculate_metric_values(){};
#endif

TLANG_NAMESPACE_END
