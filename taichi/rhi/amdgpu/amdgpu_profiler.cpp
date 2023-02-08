#include "taichi/rhi/amdgpu/amdgpu_profiler.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"

namespace taichi::lang {
#if defined(TI_WITH_AMDGPU)

std::string KernelProfilerAMDGPU::get_device_name() {
  return AMDGPUContext::get_instance().get_device_name();
}

bool KernelProfilerAMDGPU::reinit_with_metrics(
    const std::vector<std::string> metrics) {
        TI_NOT_IMPLEMENTED
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
    TI_NOT_IMPLEMENTED;
}

void KernelProfilerAMDGPU::stop(KernelProfilerBase::TaskHandle handle) {
    TI_NOT_IMPLEMENTED
}

bool KernelProfilerAMDGPU::statistics_on_traced_records() {
    TI_NOT_IMPLEMENTED
}

void KernelProfilerAMDGPU::sync() {
    amdgpuDriver::get_instance().stream_synchronize(nullptr);
}
void KernelProfilerAMDGPU::update() {
    TI_NOT_IMPLEMENTED
}

void KernelProfilerAMDGPU::clear() {
    TI_NOT_IMPLEMENTED
}

#else
std::string KernelProfilerAMDGPU::get_device_name() {
    TI_NOT_IMPLEMENTED
}

bool KernelProfilerAMDGPU::reinit_with_metrics(
    const std::vector<std::string> metrics) {
        TI_NOT_IMPLEMENTED
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

void KernelProfilerAMDGPU::clear() {
    TI_NOT_IMPLEMENTED
}

#endif

#if defined(TI_WITH_AMDGPU)

KernelProfilerBase::TaskHandle EventToolkitAMDGPU::start_with_handle(
    const std::string &kernel_name) {
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

#else

KernelProfilerBase::TaskHandle EventToolkitAMDGPU::start_with_handle(
    const std::string &kernel_name) {
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

}