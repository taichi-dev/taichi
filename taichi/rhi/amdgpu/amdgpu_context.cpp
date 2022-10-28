#define TI_RUNTIME_HOST
#include "amdgpu_context.h"

#include <unordered_map>
#include <mutex>

#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/threading.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/analysis/offline_cache_util.h"

TLANG_NAMESPACE_BEGIN

AMDGPUContext::AMDGPUContext()
    : driver_(AMDGPUDriver::get_instance_without_context()) {
  dev_count_ = 0;
  driver_.init(0);
  driver_.device_get_count(&dev_count_);
  driver_.device_get(&device_, 0);

  char name[128];
  driver_.device_get_name(name, 128, device_);

  TI_TRACE("Using AMDGPU device [id=0]: {}", name);

  driver_.context_create(&context_, 0, device_);

  const auto GB = std::pow(1024.0, 3.0);
  TI_TRACE("Total memory {:.2f} GB; free memory {:.2f} GB",
           get_total_memory() / GB, get_free_memory() / GB);

  void * hip_device_prop = std::malloc(HIP_DEVICE_PROPERTIES_STRUCT_SIZE);
  driver_.device_get_prop(hip_device_prop, device_);
  compute_capability_ = *((int *)hip_device_prop + HIP_DEVICE_GCN_ARCH);
  std::free(hip_device_prop);

  mcpu_ = fmt::format("gfx{}", compute_capability_);

  TI_TRACE("Emitting AMDGPU code for {}", mcpu_);
}

std::size_t AMDGPUContext::get_total_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&_, &ret);
  return ret;
}

std::size_t AMDGPUContext::get_free_memory() {
  std::size_t ret, _;
  driver_.mem_get_info(&ret, &_);
  return ret;
}

std::string AMDGPUContext::get_device_name() {
  constexpr uint32_t kMaxNameStringLength = 128;
  char name[kMaxNameStringLength];
  driver_.device_get_name(name, kMaxNameStringLength /*=128*/, device_);
  std::string str(name);
  return str;
}

void AMDGPUContext::launch(void *func,
                         const std::string &task_name,
                         void *arg_pointers,
                         unsigned grid_dim,
                         unsigned block_dim,
                         std::size_t dynamic_shared_mem_bytes,
                         int arg_bytes) {
  if (grid_dim > 0) {
    std::lock_guard<std::mutex> _(lock_);
    void *config[] = {(void *)0x01, const_cast<void*>(arg_pointers), 
                      (void *)0x02, &arg_bytes, (void *)0x03}; 
    driver_.launch_kernel(func, grid_dim, 1, 1, block_dim, 1, 1,
                          dynamic_shared_mem_bytes, nullptr,
                          nullptr, reinterpret_cast<void**> (&config));
  }
  if (debug_) {
    driver_.stream_synchronize(nullptr);
  }
}

AMDGPUContext::~AMDGPUContext() {
}

AMDGPUContext &AMDGPUContext::get_instance() {
  static auto context = new AMDGPUContext();
  return *context;
}

TLANG_NAMESPACE_END