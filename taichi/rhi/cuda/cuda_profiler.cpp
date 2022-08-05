#include "taichi/rhi/cuda/cuda_profiler.h"
#include "taichi/rhi/cuda/cuda_types.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)

// The init logic here is temporarily set up for test CUPTI
// will not affect default toolkit (cuEvent)
KernelProfilerCUDA::KernelProfilerCUDA(bool enable) {
  metric_list_.clear();
  if (enable) {  // default profiling toolkit: event
    tool_ = ProfilingToolkit::event;
    event_toolkit_ = std::make_unique<EventToolkit>();
  }
}

ProfilingToolkit get_toolkit_enum(std::string toolkit_name) {
  if (toolkit_name.compare("default") == 0)
    return ProfilingToolkit::event;
  else if (toolkit_name.compare("cupti") == 0)
    return ProfilingToolkit::cupti;
  else
    return ProfilingToolkit::undef;
}

bool KernelProfilerCUDA::set_profiler_toolkit(std::string toolkit_name) {
  sync();
  ProfilingToolkit set_toolkit = get_toolkit_enum(toolkit_name);
  TI_TRACE("profiler toolkit enum = {} >>> {}", tool_, set_toolkit);
  if (set_toolkit == tool_)
    return true;

  // current toolkit is CUPTI: disable
  if (tool_ == ProfilingToolkit::cupti) {
    cupti_toolkit_->end_profiling();
    cupti_toolkit_->deinit_cupti();
    cupti_toolkit_->set_status(false);
    tool_ = ProfilingToolkit::event;
    TI_TRACE("cupti >>> event ... DONE");
    return true;
  }
  // current toolkit is cuEvent: check CUPTI availability
  else if (tool_ == ProfilingToolkit::event) {
#if defined(TI_WITH_CUDA_TOOLKIT)
    if (check_cupti_availability() && check_cupti_privileges()) {
      if (cupti_toolkit_ == nullptr)
        cupti_toolkit_ = std::make_unique<CuptiToolkit>();
      cupti_toolkit_->init_cupti();
      cupti_toolkit_->begin_profiling();
      tool_ = ProfilingToolkit::cupti;
      cupti_toolkit_->set_status(true);
      TI_TRACE("event >>> cupti ... DONE");
      return true;
    }
#endif
  }
  return false;
}

std::string KernelProfilerCUDA::get_device_name() {
  return CUDAContext::get_instance().get_device_name();
}

bool KernelProfilerCUDA::reinit_with_metrics(
    const std::vector<std::string> metrics) {
  // do not pass by reference
  TI_TRACE("KernelProfilerCUDA::reinit_with_metrics");

  if (tool_ == ProfilingToolkit::event) {
    return false;
  } else if (tool_ == ProfilingToolkit::cupti) {
    cupti_toolkit_->end_profiling();
    cupti_toolkit_->deinit_cupti();
    cupti_toolkit_->reset_metrics(metrics);
    cupti_toolkit_->init_cupti();
    cupti_toolkit_->begin_profiling();
    // user selected metrics
    metric_list_.clear();
    for (auto metric : metrics)
      metric_list_.push_back(metric);
    TI_TRACE("size of metric list : {} >>> {}", metrics.size(),
             metric_list_.size());
    return true;
  }

  TI_NOT_IMPLEMENTED;
}

// deprecated, move to trace()
KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}

void KernelProfilerCUDA::trace(KernelProfilerBase::TaskHandle &task_handle,
                               const std::string &kernel_name,
                               void *kernel,
                               uint32_t grid_size,
                               uint32_t block_size,
                               uint32_t dynamic_smem_size) {
  int register_per_thread = 0;
  int static_shared_mem_per_block = 0;
  int max_active_blocks_per_multiprocessor = 0;
  CUDADriver::get_instance().kernel_get_attribute(
      &register_per_thread, CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
      kernel);
  CUDADriver::get_instance().kernel_get_attribute(
      &static_shared_mem_per_block,
      CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  CUDADriver::get_instance().kernel_get_occupancy(
      &max_active_blocks_per_multiprocessor, kernel, block_size,
      dynamic_smem_size);

  if (tool_ == ProfilingToolkit::event) {
    task_handle = event_toolkit_->start_with_handle(kernel_name);
  }
  KernelProfileTracedRecord record;
  record.name = kernel_name;
  record.register_per_thread = register_per_thread;
  record.shared_mem_per_block = static_shared_mem_per_block + dynamic_smem_size;
  record.grid_size = grid_size;
  record.block_size = block_size;
  record.active_blocks_per_multiprocessor =
      max_active_blocks_per_multiprocessor;

  traced_records_.push_back(record);
}

void KernelProfilerCUDA::stop(KernelProfilerBase::TaskHandle handle) {
  if (tool_ == ProfilingToolkit::event) {
    CUDADriver::get_instance().event_record(handle, 0);
    CUDADriver::get_instance().stream_synchronize(nullptr);

    // get elapsed time and destroy events
    auto record = event_toolkit_->get_current_event_record();
    CUDADriver::get_instance().event_elapsed_time(
        &record->kernel_elapsed_time_in_ms, record->start_event, handle);
    CUDADriver::get_instance().event_elapsed_time(
        &record->time_since_base, event_toolkit_->get_base_event(),
        record->start_event);

    CUDADriver::get_instance().event_destroy(record->start_event);
    CUDADriver::get_instance().event_destroy(record->stop_event);
  }
}

bool KernelProfilerCUDA::statistics_on_traced_records() {
  for (auto &record : traced_records_) {
    auto it =
        std::find_if(statistical_results_.begin(), statistical_results_.end(),
                     [&](KernelProfileStatisticalResult &result) {
                       return result.name == record.name;
                     });
    if (it == statistical_results_.end()) {
      statistical_results_.emplace_back(record.name);
      it = std::prev(statistical_results_.end());
    }
    it->insert_record(record.kernel_elapsed_time_in_ms);
    total_time_ms_ += record.kernel_elapsed_time_in_ms;
  }

  return true;
}

void KernelProfilerCUDA::sync() {
  CUDADriver::get_instance().stream_synchronize(nullptr);
}

void KernelProfilerCUDA::update() {
  if (tool_ == ProfilingToolkit::event) {
    event_toolkit_->update_record(records_size_after_sync_, traced_records_);
    event_toolkit_->update_timeline(traced_records_);
    statistics_on_traced_records();  // TODO: deprecated
    event_toolkit_->clear();
  } else if (tool_ == ProfilingToolkit::cupti) {
    cupti_toolkit_->update_record(records_size_after_sync_, traced_records_);
    statistics_on_traced_records();  // TODO: deprecated
    this->reinit_with_metrics(metric_list_);
  }

  records_size_after_sync_ = traced_records_.size();
}

void KernelProfilerCUDA::clear() {
  // sync(); //decoupled: trigger from the foront end
  update();
  total_time_ms_ = 0;
  records_size_after_sync_ = 0;
  traced_records_.clear();
  statistical_results_.clear();
}

// must be called immediately after KernelProfilerCUDA::trace()
bool KernelProfilerCUDA::record_kernel_attributes(void *kernel,
                                                  uint32_t grid_size,
                                                  uint32_t block_size,
                                                  uint32_t dynamic_smem_size) {
  int register_per_thread = 0;
  int static_shared_mem_per_block = 0;
  int max_active_blocks_per_multiprocessor = 0;

  CUDADriver::get_instance().kernel_get_attribute(
      &register_per_thread, CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS,
      kernel);
  CUDADriver::get_instance().kernel_get_attribute(
      &static_shared_mem_per_block,
      CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  CUDADriver::get_instance().kernel_get_occupancy(
      &max_active_blocks_per_multiprocessor, kernel, block_size,
      dynamic_smem_size);

  KernelProfileTracedRecord &traced_record = traced_records_.back();
  traced_record.register_per_thread = register_per_thread;
  traced_record.shared_mem_per_block =
      static_shared_mem_per_block + dynamic_smem_size;
  traced_record.grid_size = grid_size;
  traced_record.block_size = block_size;
  traced_record.active_blocks_per_multiprocessor =
      max_active_blocks_per_multiprocessor;

  return true;
}

#else

KernelProfilerCUDA::KernelProfilerCUDA(bool enable) {
  TI_NOT_IMPLEMENTED;
}
std::string KernelProfilerCUDA::get_device_name() {
  TI_NOT_IMPLEMENTED;
}
bool KernelProfilerCUDA::reinit_with_metrics(
    const std::vector<std::string> metrics) {
  return false;  // public API for all backend, do not use TI_NOT_IMPLEMENTED;
}
KernelProfilerBase::TaskHandle KernelProfilerCUDA::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::trace(KernelProfilerBase::TaskHandle &task_handle,
                               const std::string &kernel_name,
                               void *kernel,
                               uint32_t grid_size,
                               uint32_t block_size,
                               uint32_t dynamic_smem_size) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::stop(KernelProfilerBase::TaskHandle handle) {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::sync() {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::update() {
  TI_NOT_IMPLEMENTED;
}
void KernelProfilerCUDA::clear() {
  TI_NOT_IMPLEMENTED;
}
bool KernelProfilerCUDA::record_kernel_attributes(void *kernel,
                                                  uint32_t grid_size,
                                                  uint32_t block_size,
                                                  uint32_t dynamic_smem_size) {
  TI_NOT_IMPLEMENTED;
}
#endif

// default profiling toolkit : cuEvent
// for now put it together with KernelProfilerCUDA
#if defined(TI_WITH_CUDA)
KernelProfilerBase::TaskHandle EventToolkit::start_with_handle(
    const std::string &kernel_name) {
  EventRecord record;
  record.name = kernel_name;

  CUDADriver::get_instance().event_create(&(record.start_event),
                                          CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_create(&(record.stop_event),
                                          CU_EVENT_DEFAULT);
  CUDADriver::get_instance().event_record((record.start_event), 0);
  event_records_.push_back(record);

  if (!base_event_) {
    // Note that CUDA driver API only allows querying relative time difference
    // between two events, therefore we need to manually build a mapping
    // between GPU and CPU time.
    // TODO: periodically reinitialize for more accuracy.
    int n_iters = 100;
    // Warm up CUDA driver, and use the final event as the base event.
    for (int i = 0; i < n_iters; i++) {
      void *e;
      CUDADriver::get_instance().event_create(&e, CU_EVENT_DEFAULT);
      CUDADriver::get_instance().event_record(e, 0);
      CUDADriver::get_instance().event_synchronize(e);
      auto final_t = Time::get_time();
      if (i == n_iters - 1) {
        base_event_ = e;
        // TODO: figure out a better way to synchronize CPU and GPU time.
        constexpr float64 cuda_time_offset = 3e-4;
        // Since event recording and synchronization can take 5 us, it's hard
        // to exactly measure the real event time. Also note there seems to be
        // a systematic time offset on CUDA, so adjust for that using a 3e-4 s
        // magic number.
        base_time_ = final_t + cuda_time_offset;
      } else {
        CUDADriver::get_instance().event_destroy(e);
      }
    }
  }
  return record.stop_event;
}

void EventToolkit::update_record(
    uint32_t records_size_after_sync,
    std::vector<KernelProfileTracedRecord> &traced_records) {
  uint32_t events_num = event_records_.size();
  uint32_t records_num = traced_records.size();
  TI_ERROR_IF(records_size_after_sync + events_num != records_num,
              "KernelProfilerCUDA::EventToolkit: event_records_.size({}) != "
              "traced_records_.size({})",
              records_size_after_sync + events_num, records_num);

  uint32_t idx = 0;
  for (auto &record : event_records_) {
    // copy to traced_records_ then clear event_records_
    traced_records[records_size_after_sync + idx].kernel_elapsed_time_in_ms =
        record.kernel_elapsed_time_in_ms;
    traced_records[records_size_after_sync + idx].time_since_base =
        record.time_since_base;
    idx++;
  }
}

void EventToolkit::update_timeline(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  if (Timelines::get_instance().get_enabled()) {
    auto &timeline = Timeline::get_this_thread_instance();
    for (auto &record : traced_records) {
      // param of insert_event() :
      // struct TimelineEvent @ taichi/taichi/system/timeline.h
      timeline.insert_event({record.name, /*param_name=begin*/ true,
                             base_time_ + record.time_since_base * 1e-3,
                             "cuda"});
      timeline.insert_event({record.name, /*param_name=begin*/ false,
                             base_time_ + (record.time_since_base +
                                           record.kernel_elapsed_time_in_ms) *
                                              1e-3,
                             "cuda"});
    }
  }
}

#else
KernelProfilerBase::TaskHandle EventToolkit::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}
void EventToolkit::update_record(
    uint32_t records_size_after_sync,
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
void EventToolkit::update_timeline(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
#endif

TLANG_NAMESPACE_END
