
#include "taichi/backends/cuda/cupti_toolkit.h"

TLANG_NAMESPACE_BEGIN

void CUPTIToolkit::set_mode(KernelProfilingMode mode) {
  cupti_config_.profiling_mode = mode;
}

CUPTIToolkit::CUPTIToolkit(KernelProfilingMode mode) {
  TI_TRACE("CUPTIToolkit::CUPTIToolkit() ");
  set_mode(mode);
}

CUPTIToolkit::~CUPTIToolkit() {
  end_profiling();
  deinit_cupti();
}

void CUPTIToolkit::record_launched_kernel(std::string name) {
  KernelProfileTracedRecord record;
  record.kernel_name = name;
  traced_records_.push_back(record);
}

bool CUPTIToolkit::statistics_on_traced_records(
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

#if defined(TI_WITH_TOOLKIT_CUDA)
    if (cupti_config_.profiling_mode == KernelProfilingMode::cupti_onepass) {
      it->cuda_mem_access(
          traced_records_[resultIndex].kernel_gloabl_load_byets,
          traced_records_[resultIndex].kernel_gloabl_store_byets);
    }
    if (cupti_config_.profiling_mode == KernelProfilingMode::cupti_detailed) {
      it->cuda_mem_access(
          traced_records_[resultIndex].kernel_gloabl_load_byets,
          traced_records_[resultIndex].kernel_gloabl_store_byets);
      it->cuda_utilization_ratio(
          traced_records_[resultIndex].utilization_ratio_sm,
          traced_records_[resultIndex].utilization_ratio_mem);
    }
#endif
  }

  return true;
}

void CUPTIToolkit::clear_traced_records() {
  traced_records_.clear();
}

#if defined(TI_WITH_TOOLKIT_CUDA)
// TODO : CUPTI_PROFILER
#else
bool CUPTIToolkit::init_cupti(){
  TI_NOT_IMPLEMENTED;
  return false;
}
bool CUPTIToolkit::begin_profiling(){
  TI_NOT_IMPLEMENTED;
  return false;
}
bool CUPTIToolkit::end_profiling(){
  TI_NOT_IMPLEMENTED;
  return false;
}
bool CUPTIToolkit::deinit_cupti(){
  TI_NOT_IMPLEMENTED;
  return false;
}
bool CUPTIToolkit::calculate_metric_values(){
  TI_NOT_IMPLEMENTED;
  return false;
}
#endif

TLANG_NAMESPACE_END
