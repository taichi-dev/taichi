#include "taichi/rhi/amdgpu/amdgpu_profiler.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/rhi/amdgpu/amdgpu_types.h"

namespace taichi::lang {
#if defined(TI_WITH_AMDGPU)

std::string KernelProfilerAMDGPU::get_device_name() {
  return AMDGPUContext::get_instance().get_device_name();
}

bool KernelProfilerAMDGPU::reinit_with_metrics(
    const std::vector<std::string> metrics) {
  TI_NOT_IMPLEMENTED
}

bool KernelProfilerAMDGPU::set_profiler_toolkit(std::string toolkit_name) {
  if (toolkit_name.compare("default") == 0) {
    return true;
  }
  TI_WARN("Only default(event) profiler is allowed on AMDGPU");
  return false;
}

KernelProfilerBase::TaskHandle KernelProfilerAMDGPU::start_with_handle(
    const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}

void KernelProfilerAMDGPU::trace(KernelProfilerBase::TaskHandle &task_handle,
                                 const std::string &kernel_name,
                                 void *kernel,
                                 uint32_t grid_size,
                                 uint32_t block_size,
                                 uint32_t dynamic_smem_size) {
  int register_per_thread = 0;
  int static_shared_mem_per_block = 0;
  // int max_active_blocks_per_multiprocessor = 0;
  task_handle = event_toolkit_->start_with_handle(kernel_name);
  KernelProfileTracedRecord record;

  AMDGPUDriver::get_instance().kernel_get_attribute(
      &register_per_thread, HIPfunction_attribute::HIP_FUNC_ATTRIBUTE_NUM_REGS,
      kernel);
  AMDGPUDriver::get_instance().kernel_get_attribute(
      &static_shared_mem_per_block,
      HIPfunction_attribute::HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel);
  // kernel_get_occupancy doesn't work well
  // AMDGPUDriver::get_instance().kernel_get_occupancy(
  //     &max_active_blocks_per_multiprocessor, kernel, block_size,
  //     dynamic_smem_size);

  record.name = kernel_name;
  record.register_per_thread = register_per_thread;
  record.shared_mem_per_block = static_shared_mem_per_block + dynamic_smem_size;
  record.grid_size = grid_size;
  record.block_size = block_size;
  // record.active_blocks_per_multiprocessor =
  //    max_active_blocks_per_multiprocessor;

  traced_records_.push_back(record);
}

void KernelProfilerAMDGPU::stop(KernelProfilerBase::TaskHandle handle) {
  AMDGPUDriver::get_instance().event_record(handle, 0);
  AMDGPUDriver::get_instance().stream_synchronize(nullptr);

  // get elapsed time and destroy events
  auto record = event_toolkit_->get_current_event_record();
  AMDGPUDriver::get_instance().event_elapsed_time(
      &record->kernel_elapsed_time_in_ms, record->start_event, handle);
  AMDGPUDriver::get_instance().event_elapsed_time(
      &record->time_since_base, event_toolkit_->get_base_event(),
      record->start_event);

  AMDGPUDriver::get_instance().event_destroy(record->start_event);
  AMDGPUDriver::get_instance().event_destroy(record->stop_event);
}

bool KernelProfilerAMDGPU::statistics_on_traced_records() {
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

void KernelProfilerAMDGPU::sync() {
  AMDGPUDriver::get_instance().stream_synchronize(nullptr);
}

void KernelProfilerAMDGPU::update() {
  event_toolkit_->update_record(records_size_after_sync_, traced_records_);
  event_toolkit_->update_timeline(traced_records_);
  statistics_on_traced_records();
  event_toolkit_->clear();
  records_size_after_sync_ = traced_records_.size();
}

void KernelProfilerAMDGPU::clear() {
  update();
  total_time_ms_ = 0;
  records_size_after_sync_ = 0;
  traced_records_.clear();
  statistical_results_.clear();
}

#else
std::string KernelProfilerAMDGPU::get_device_name() {
  TI_NOT_IMPLEMENTED
}

bool KernelProfilerAMDGPU::reinit_with_metrics(
    const std::vector<std::string> metrics){TI_NOT_IMPLEMENTED}

KernelProfilerBase::TaskHandle
    KernelProfilerAMDGPU::start_with_handle(const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}

void KernelProfilerAMDGPU::trace(KernelProfilerBase::TaskHandle &task_handle,
                                 const std::string &kernel_name,
                                 void *kernel,
                                 uint32_t grid_size,
                                 uint32_t block_size,
                                 uint32_t dynamic_smem_size) {
  TI_NOT_IMPLEMENTED;
}

void KernelProfilerAMDGPU::stop(KernelProfilerBase::TaskHandle handle) {
  TI_NOT_IMPLEMENTED
}

bool KernelProfilerAMDGPU::statistics_on_traced_records() {
  TI_NOT_IMPLEMENTED
}

void KernelProfilerAMDGPU::sync() {
  TI_NOT_IMPLEMENTED
}
void KernelProfilerAMDGPU::update() {
  TI_NOT_IMPLEMENTED
}

void KernelProfilerAMDGPU::clear(){TI_NOT_IMPLEMENTED}

#endif

#if defined(TI_WITH_AMDGPU)

KernelProfilerBase::TaskHandle EventToolkitAMDGPU::start_with_handle(
    const std::string &kernel_name) {
  EventRecord record;
  record.name = kernel_name;

  AMDGPUDriver::get_instance().event_create(&(record.start_event),
                                            HIP_EVENT_DEFAULT);
  AMDGPUDriver::get_instance().event_create(&(record.stop_event),
                                            HIP_EVENT_DEFAULT);
  AMDGPUDriver::get_instance().event_record((record.start_event), 0);
  event_records_.push_back(record);

  if (!base_event_) {
    int n_iters = 100;
    // Warm up
    for (int i = 0; i < n_iters; i++) {
      void *e;
      AMDGPUDriver::get_instance().event_create(&e, HIP_EVENT_DEFAULT);
      AMDGPUDriver::get_instance().event_record(e, 0);
      AMDGPUDriver::get_instance().event_synchronize(e);
      auto final_t = Time::get_time();
      if (i == n_iters - 1) {
        base_event_ = e;
        // ignore the overhead of sync, event_create and systematic time offset.
        base_time_ = final_t;
      } else {
        AMDGPUDriver::get_instance().event_destroy(e);
      }
    }
  }
  return record.stop_event;
}

void EventToolkitAMDGPU::update_record(
    uint32_t records_size_after_sync,
    std::vector<KernelProfileTracedRecord> &traced_records) {
  uint32_t events_num = event_records_.size();
  uint32_t records_num = traced_records.size();
  TI_ERROR_IF(
      records_size_after_sync + events_num != records_num,
      "KernelProfilerAMDGPU::EventToolkitAMDGPU: event_records_.size({}) != "
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

void EventToolkitAMDGPU::update_timeline(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  if (Timelines::get_instance().get_enabled()) {
    auto &timeline = Timeline::get_this_thread_instance();
    for (auto &record : traced_records) {
      timeline.insert_event({record.name, /*param_name=begin*/ true,
                             base_time_ + record.time_since_base * 1e-3,
                             "amdgpu"});
      timeline.insert_event({record.name, /*param_name=begin*/ false,
                             base_time_ + (record.time_since_base +
                                           record.kernel_elapsed_time_in_ms) *
                                              1e-3,
                             "amdgpu"});
    }
  }
}

#else

KernelProfilerBase::TaskHandle
    EventToolkitAMDGPU::start_with_handle(const std::string &kernel_name) {
  TI_NOT_IMPLEMENTED;
}
void EventToolkitAMDGPU::update_record(
    uint32_t records_size_after_sync,
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}
void EventToolkitAMDGPU::update_timeline(
    std::vector<KernelProfileTracedRecord> &traced_records) {
  TI_NOT_IMPLEMENTED;
}

#endif

}  // namespace taichi::lang
