#include "taichi/backends/cuda/cuda_profiler.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"


TLANG_NAMESPACE_BEGIN

CUDAProfiler::CUDAProfiler() 
{
  TI_TRACE("CUDAProfiler::CUDAProfiler() ");
}

CUDAProfiler::~CUDAProfiler() {
  if( profiler_config_.profiler_type == CUDA_KERNEL_PROFILER_CUPTI ){
    end_profiling();
    deinit_cupti();
  }
}


bool CUDAProfiler::set_profiling_mode(KernelProfilerMode profiling_mode){

  bool enable = profiling_mode == KernelProfilerMode::enable
              | profiling_mode == KernelProfilerMode::cuda_accurate
              | profiling_mode == KernelProfilerMode::cuda_detailed;
  if(!enable){
    return false;
  }

  CUDAKernalProfiler profiler_type = (profiling_mode == KernelProfilerMode::cuda_accurate) 
                                  || (profiling_mode == KernelProfilerMode::cuda_detailed)
                                   ? CUDA_KERNEL_PROFILER_CUPTI : CUDA_KERNEL_PROFILER_EVENT;
  profiler_config_.profiling_mode = profiling_mode;
  
  if(enable && profiler_type == CUDA_KERNEL_PROFILER_EVENT){
    profiler_config_.enable = true;
    profiler_config_.profiler_type = CUDA_KERNEL_PROFILER_EVENT;
    profiler_config_.profiling_mode = KernelProfilerMode::enable;
    TI_TRACE("CUDA_KERNEL_PROFILER_EVENT : enable");
    TI_TRACE("CUDAProfiler enable = {}",profiler_config_.enable);
    TI_TRACE("profiler_type : {}",profiler_config_.profiler_type);
    return true;
  }

#if !defined (TI_WITH_TOOLKIT_CUDA)
  if(enable && profiler_type == CUDA_KERNEL_PROFILER_CUPTI){
    TI_WARN("CUPTI toolkit is not compiled with taichi, fallback to cuEvent kernel profiler");
    TI_WARN("to use CUPTI kernel profiler : TAICHI_CMAKE_ARGS=-DTI_WITH_TOOLKIT_CUDA=True python3 setup.py develop --user");
    
    profiler_config_.enable = true;
    profiler_config_.profiler_type = CUDA_KERNEL_PROFILER_EVENT;
    profiler_config_.profiling_mode = KernelProfilerMode::enable;

    TI_TRACE("CUDA_KERNEL_PROFILER_EVENT : enable");
    TI_TRACE("CUDAProfiler enable = {}",profiler_config_.enable);
    TI_TRACE("profiler_type : {}",profiler_config_.profiler_type);
    return true;
  }
#else
//TODO::CUPTI_PROFILER
#endif
}


bool CUDAProfiler::statisticsOnRecords(std::vector<KernelProfileRecord> &records, double &total_time_ms) {

  size_t result_nums = traced_records_.size();
  if (!result_nums) {
    TI_WARN("traced_records_ is empty!");
    return false;
  }

  for (size_t resultIndex = 0; resultIndex < result_nums; ++resultIndex) {
    auto it = std::find_if(
      records.begin(), records.end(),
      [&](KernelProfileRecord &r) { return r.name == traced_records_[resultIndex].kernel_name; });
    if (it == records.end()) {
      records.push_back(traced_records_[resultIndex].kernel_name);
      it = std::prev(records.end());
    }
    it->insert_sample(traced_records_[resultIndex].kernel_elapsed_time_in_ms);
    total_time_ms += traced_records_[resultIndex].kernel_elapsed_time_in_ms;

    if(profiler_config_.profiling_mode == KernelProfilerMode::cuda_accurate){
      it->cuda_global_access(traced_records_[resultIndex].kernel_gloabl_load_byets, traced_records_[resultIndex].kernel_gloabl_store_byets);
    }
    if(profiler_config_.profiling_mode == KernelProfilerMode::cuda_detailed){
      it->cuda_global_access(traced_records_[resultIndex].kernel_gloabl_load_byets, traced_records_[resultIndex].kernel_gloabl_store_byets);
      it->cuda_uti_ratio(traced_records_[resultIndex].utilization_ratio_sm, traced_records_[resultIndex].utilization_ratio_mem);
    }
  }

  return true;
}

CUDAProfiler &CUDAProfiler::get_instance_without_context() {
  static CUDAProfiler *instance = new CUDAProfiler();
  return *instance;
}

CUDAProfiler &CUDAProfiler::get_instance() {
  CUDAContext::get_instance();
  return get_instance_without_context();
}

bool CUDAProfiler::printTracedRecords() {

  size_t result_nums = traced_records_.size();
  if (!result_nums) {
    TI_WARN("traced_records_ is empty!");
    return false;
  }

  for (size_t resultIndex = 0; resultIndex < result_nums; ++resultIndex) {
    TI_INFO("Kernel: name[{}] time[{:6.6f}]ms ldg[{}]bytes stg[{}]bytes", 
      traced_records_[resultIndex].kernel_name, 
      traced_records_[resultIndex].kernel_elapsed_time_in_ms,
      traced_records_[resultIndex].kernel_gloabl_load_byets,
      traced_records_[resultIndex].kernel_gloabl_store_byets);
    if(profiler_config_.profiling_mode == KernelProfilerMode::cuda_detailed){
      TI_INFO("        detailed metrics: sm.utilization.ratio[{:7.2f}] mem.utilization.ratio[{:7.2f}]", 
      traced_records_[resultIndex].utilization_ratio_sm, 
      traced_records_[resultIndex].utilization_ratio_mem);
    }
  }

  return true;
}

#if defined(TI_WITH_TOOLKIT_CUDA)
//TODO::CUPTI_PROFILER
#else
bool CUDAProfiler::init_cupti(){};
bool CUDAProfiler::begin_profiling(){};
bool CUDAProfiler::end_profiling(){};
bool CUDAProfiler::deinit_cupti(){};
bool CUDAProfiler::traceMetricValues(){};
#endif

TLANG_NAMESPACE_END