#include "taichi/rhi/flagos/flagos_device.h"
#include "taichi/rhi/llvm/device_memory_pool.h"

#include "taichi/jit/jit_module.h"
#include "taichi/common/logging.h"

namespace taichi {
namespace lang {

namespace flagos {

// FlagOS context wrapper implementation
struct FlagosDevice::FlagosContextWrapper {
  // Placeholder for actual FlagOS context
  // This will be replaced with actual FlagOS SDK calls
  bool initialized{false};

  void initialize(const std::string &chip_name) {
    TI_INFO("Initializing FlagOS backend with chip: {}", chip_name);
    // TODO: Call FlagOS SDK initialization
    // flagos_init(chip_name.c_str());
    initialized = true;
  }

  void *allocate_memory(size_t size, bool managed) {
    // TODO: Use FlagOS memory allocation
    // return flagos_malloc(size, managed);
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  void free_memory(void *ptr) {
    // TODO: Use FlagOS memory deallocation
    // flagos_free(ptr);
  }

  void memcpy_host_to_device(void *dst, const void *src, size_t size) {
    // TODO: Use FlagOS H2D copy
    TI_NOT_IMPLEMENTED;
  }

  void memcpy_device_to_host(void *dst, const void *src, size_t size) {
    // TODO: Use FlagOS D2H copy
    TI_NOT_IMPLEMENTED;
  }

  void memcpy_device_to_device(void *dst, const void *src, size_t size) {
    // TODO: Use FlagOS D2D copy
    TI_NOT_IMPLEMENTED;
  }

  void memset(void *ptr, int value, size_t size) {
    // TODO: Use FlagOS memset
    TI_NOT_IMPLEMENTED;
  }

  void synchronize() {
    // TODO: Use FlagOS synchronization
    TI_NOT_IMPLEMENTED;
  }

  size_t get_total_memory() {
    // TODO: Query FlagOS device memory
    return 0;
  }

  void launch_kernel(const std::string &kernel_name,
                     void **args,
                     uint32_t grid_dim,
                     uint32_t block_dim) {
    // TODO: Launch kernel via FlagOS
    TI_NOT_IMPLEMENTED;
  }
};

FlagosDevice::FlagosDevice()
    : context_(std::make_unique<FlagosContextWrapper>()) {
  // Initialize the device memory pool
  DeviceMemoryPool::get_instance(false /*merge_upon_release*/);

  // Get target chip from environment variable or use default
  const char *chip_env = std::getenv("TI_FLAGOS_CHIP");
  if (chip_env) {
    target_chip_ = chip_env;
  }

  TI_INFO("FlagOS Device created, target chip: {}", target_chip_);
}

FlagosDevice::~FlagosDevice() {
  clear();
}

void FlagosDevice::initialize(const std::string &chip_name) {
  target_chip_ = chip_name;
  context_->initialize(chip_name);
}

FlagosDevice::AllocInfo FlagosDevice::get_alloc_info(
    const DeviceAllocation handle) {
  validate_device_alloc(handle);
  return allocations_[handle.alloc_id];
}

RhiResult FlagosDevice::allocate_memory(const AllocParams &params,
                                        DeviceAllocation *out_devalloc) {
  AllocInfo info;

  auto &mem_pool = DeviceMemoryPool::get_instance();

  bool managed = params.host_read || params.host_write;

  // Try to use FlagOS allocation first, fallback to memory pool
  void *ptr = nullptr;
  if (context_->initialized) {
    ptr = context_->allocate_memory(params.size, managed);
  }

  if (ptr == nullptr) {
    ptr = mem_pool.allocate(params.size, DeviceMemoryPool::page_size, managed);
  }

  if (ptr == nullptr) {
    return RhiResult::out_of_memory;
  }

  info.ptr = ptr;
  info.size = params.size;
  info.is_imported = false;
  info.use_cached = false;
  info.use_preallocated = false;

  if (info.ptr == nullptr) {
    return RhiResult::out_of_memory;
  }

  // Initialize memory to zero
  if (context_->initialized) {
    context_->memset(info.ptr, 0, info.size);
  } else {
    std::memset(info.ptr, 0, info.size);
  }

  *out_devalloc = DeviceAllocation{};
  out_devalloc->alloc_id = allocations_.size();
  out_devalloc->device = this;

  allocations_.push_back(info);
  return RhiResult::success;
}

DeviceAllocation FlagosDevice::allocate_memory_runtime(
    const LlvmRuntimeAllocParams &params) {
  AllocInfo info;
  info.size = taichi::iroundup(params.size, taichi_page_size);
  if (params.host_read || params.host_write) {
    TI_NOT_IMPLEMENTED
  } else {
    info.ptr =
        DeviceMemoryPool::get_instance().allocate_with_cache(this, params);
    TI_ASSERT(info.ptr != nullptr);

    if (context_->initialized) {
      context_->memset(info.ptr, 0, info.size);
    }
  }
  info.is_imported = false;
  info.use_cached = true;
  info.use_preallocated = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

uint64_t *FlagosDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size, params.result_buffer);

  wait_idle();

  uint64 *ret{nullptr};
  // Read back the result
  if (context_->initialized) {
    context_->memcpy_device_to_host(&ret, params.result_buffer, sizeof(uint64));
  }
  return ret;
}

void FlagosDevice::dealloc_memory(DeviceAllocation handle) {
  // After reset, all allocations are invalid
  if (allocations_.empty()) {
    return;
  }

  validate_device_alloc(handle);
  AllocInfo &info = allocations_[handle.alloc_id];
  if (info.ptr == nullptr) {
    TI_ERROR("the DeviceAllocation is already deallocated");
  }
  TI_ASSERT(!info.is_imported);
  if (info.use_cached) {
    DeviceMemoryPool::get_instance().release(info.size, (uint64_t *)info.ptr,
                                             false);
  } else if (!info.use_preallocated) {
    if (context_->initialized) {
      context_->free_memory(info.ptr);
    } else {
      DeviceMemoryPool::get_instance().release(info.size, info.ptr);
    }
  }
  info.ptr = nullptr;
}

RhiResult FlagosDevice::map(DeviceAllocation alloc, void **mapped_ptr) {
  AllocInfo &info = allocations_[alloc.alloc_id];
  size_t size = info.size;
  info.mapped = new char[size];

  if (context_->initialized) {
    context_->memcpy_device_to_host(info.mapped, info.ptr, size);
  }

  *mapped_ptr = info.mapped;
  return RhiResult::success;
}

void FlagosDevice::unmap(DeviceAllocation alloc) {
  AllocInfo &info = allocations_[alloc.alloc_id];

  if (context_->initialized) {
    context_->memcpy_host_to_device(info.ptr, info.mapped, info.size);
  }

  delete[] static_cast<char *>(info.mapped);
  info.mapped = nullptr;
}

void FlagosDevice::memcpy_internal(DevicePtr dst,
                                   DevicePtr src,
                                   uint64_t size) {
  void *dst_ptr =
      static_cast<char *>(allocations_[dst.alloc_id].ptr) + dst.offset;
  void *src_ptr =
      static_cast<char *>(allocations_[src.alloc_id].ptr) + src.offset;

  if (context_->initialized) {
    context_->memcpy_device_to_device(dst_ptr, src_ptr, size);
  }
}

DeviceAllocation FlagosDevice::import_memory(void *ptr, size_t size) {
  AllocInfo info;
  info.ptr = ptr;
  info.size = size;
  info.is_imported = true;

  DeviceAllocation alloc;
  alloc.alloc_id = allocations_.size();
  alloc.device = this;

  allocations_.push_back(info);
  return alloc;
}

std::size_t FlagosDevice::get_total_memory() {
  if (context_->initialized) {
    return context_->get_total_memory();
  }
  return 0;
}

void FlagosDevice::wait_idle() {
  if (context_->initialized) {
    context_->synchronize();
  }
}

void FlagosDevice::clear() {
  allocations_.clear();
  context_->initialized = false;
}

RhiResult FlagosDevice::upload_data(DevicePtr *device_ptr,
                                    const void **data,
                                    size_t *size,
                                    int num_alloc) noexcept {
  for (int i = 0; i < num_alloc; i++) {
    void *dst_ptr =
        static_cast<char *>(allocations_[device_ptr[i].alloc_id].ptr) +
        device_ptr[i].offset;
    if (context_->initialized) {
      context_->memcpy_host_to_device(dst_ptr, data[i], size[i]);
    }
  }
  return RhiResult::success;
}

RhiResult FlagosDevice::readback_data(
    DevicePtr *device_ptr,
    void **data,
    size_t *size,
    int num_alloc,
    const std::vector<StreamSemaphore> &wait_sema) noexcept {
  for (int i = 0; i < num_alloc; i++) {
    void *src_ptr =
        static_cast<char *>(allocations_[device_ptr[i].alloc_id].ptr) +
        device_ptr[i].offset;
    if (context_->initialized) {
      context_->memcpy_device_to_host(data[i], src_ptr, size[i]);
    }
  }
  return RhiResult::success;
}

void FlagosDevice::launch_kernel(const std::string &kernel_name,
                                 void **args,
                                 uint32_t grid_dim,
                                 uint32_t block_dim) {
  if (context_->initialized) {
    context_->launch_kernel(kernel_name, args, grid_dim, block_dim);
  } else {
    TI_ERROR("FlagOS context not initialized");
  }
}

bool FlagosDevice::is_chip_supported(const std::string &chip_name) {
  // TODO: Query FlagOS for supported chips
  static const std::set<std::string> supported_chips = {
      "mlu370", "mlu590", "ascend910", "ascend310", "dcu", "gcu", "generic"};
  return supported_chips.count(chip_name) > 0;
}

std::vector<std::string> FlagosDevice::get_supported_chips() {
  // TODO: Query FlagOS for available chips
  return {"mlu370", "mlu590", "ascend910", "ascend310",
          "dcu",    "gcu",    "generic"};
}

}  // namespace flagos

}  // namespace lang

}  // namespace taichi
