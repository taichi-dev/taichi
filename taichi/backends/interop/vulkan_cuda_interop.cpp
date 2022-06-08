#include "taichi/backends/interop/vulkan_cuda_interop.h"

#if TI_WITH_VULKAN && TI_WITH_CUDA
#include "taichi/backends/cuda/cuda_device.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/runtime/vulkan/vulkan_device.h"
#endif  // TI_WITH_VULKAN && TI_WITH_CUDA

#include <unordered_map>

namespace taichi {
namespace lang {

#if TI_WITH_VULKAN && TI_WITH_CUDA

using namespace taichi::lang::vulkan;
using namespace taichi::lang::cuda;

namespace {

#ifdef _WIN64  // For windows
HANDLE get_device_mem_handle(VkDeviceMemory &mem, VkDevice device) {
  HANDLE handle;

  VkMemoryGetWin32HandleInfoKHR memory_get_win32_handle_info = {};
  memory_get_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  memory_get_win32_handle_info.pNext = nullptr;
  memory_get_win32_handle_info.memory = mem;
  memory_get_win32_handle_info.handleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  auto fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
          device, "vkGetMemoryWin32HandleKHR");

  if (fpGetMemoryWin32HandleKHR == nullptr) {
    TI_ERROR("vkGetMemoryFdKHR is nullptr");
  }

  auto result =
      fpGetMemoryWin32HandleKHR(device, &memory_get_win32_handle_info, &handle);
  if (result != VK_SUCCESS) {
    TI_ERROR("vkGetMemoryWin32HandleKHR failed");
  }

  return handle;
}
#else
int get_device_mem_handle(VkDeviceMemory &mem, VkDevice device) {
  int fd;

  VkMemoryGetFdInfoKHR memory_get_fd_info = {};
  memory_get_fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  memory_get_fd_info.pNext = nullptr;
  memory_get_fd_info.memory = mem;
  memory_get_fd_info.handleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

  auto fpGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");

  if (fpGetMemoryFdKHR == nullptr) {
    TI_ERROR("vkGetMemoryFdKHR is nullptr");
  }
  fpGetMemoryFdKHR(device, &memory_get_fd_info, &fd);

  return fd;
}
#endif

#ifdef _WIN64
CUexternalMemory import_vk_memory_object_from_handle(HANDLE handle,
                                                     unsigned long long size,
                                                     bool is_dedicated) {
  CUexternalMemory ext_mem = nullptr;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
  desc.handle.win32.handle = handle;
  desc.size = size;
  if (is_dedicated) {
    desc.flags |= CUDA_EXTERNAL_MEMORY_DEDICATED;
  }

  CUDADriver::get_instance().import_external_memory(&ext_mem, &desc);
  return ext_mem;
}
#else
CUexternalMemory import_vk_memory_object_from_handle(int fd,
                                                     unsigned long long size,
                                                     bool is_dedicated) {
  CUexternalMemory ext_mem = nullptr;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  desc.handle.fd = fd;
  desc.size = size;
  if (is_dedicated) {
    desc.flags |= CUDA_EXTERNAL_MEMORY_DEDICATED;
  }
  CUDADriver::get_instance().import_external_memory(&ext_mem, &desc);
  return ext_mem;
}
#endif

void *map_buffer_onto_external_memory(CUexternalMemory ext_mem,
                                      unsigned long long offset,
                                      unsigned long long size) {
  void *ptr = nullptr;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.offset = offset;
  desc.size = size;

  CUDADriver::get_instance().external_memory_get_mapped_buffer(
      (CUdeviceptr *)&ptr, ext_mem, &desc);
  return ptr;
}

void *get_cuda_memory_pointer(VkDeviceMemory mem,
                              VkDeviceSize mem_size,
                              VkDeviceSize offset,
                              VkDeviceSize buffer_size,
                              VkDevice device) {
  auto handle = get_device_mem_handle(mem, device);
  CUexternalMemory externalMem =
      import_vk_memory_object_from_handle(handle, mem_size, false);
  return map_buffer_onto_external_memory(externalMem, offset, buffer_size);
}

void cuda_memcpy(void *dst, void *src, size_t size) {
  CUDADriver::get_instance().memcpy_device_to_device(dst, src, size);
}

}  // namespace

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size) {
  VulkanDevice *vk_dev = dynamic_cast<VulkanDevice *>(dst.device);
  CudaDevice *cuda_dev = dynamic_cast<CudaDevice *>(src.device);

  DeviceAllocation dst_alloc(dst);
  DeviceAllocation src_alloc(src);

  static std::unordered_map<
      VulkanDevice *,
      std::unordered_map<CudaDevice *,
                         std::unordered_map<int, unsigned char *>>>
      alloc_base_ptrs_all;
  std::unordered_map<int, unsigned char *> &alloc_base_ptrs =
      alloc_base_ptrs_all[vk_dev][cuda_dev];

  if (alloc_base_ptrs.find(dst_alloc.alloc_id) == alloc_base_ptrs.end()) {
    auto [base_mem, alloc_offset, alloc_size] =
        vk_dev->get_vkmemory_offset_size(dst_alloc);
    // this might be smaller than the actual size of the VkDeviceMemory, but it
    // is big enough to cover the region of this buffer, so it's fine.
    size_t mem_size = alloc_offset + alloc_size;
    void *alloc_base_ptr = get_cuda_memory_pointer(
        base_mem, /*mem_size=*/mem_size, /*offset=*/alloc_offset,
        /*buffer_size=*/alloc_size, vk_dev->vk_device());
    alloc_base_ptrs[dst_alloc.alloc_id] = (unsigned char *)alloc_base_ptr;
  }

  unsigned char *dst_cuda_ptr =
      alloc_base_ptrs.at(dst_alloc.alloc_id) + dst.offset;

  CudaDevice::AllocInfo src_alloc_info = cuda_dev->get_alloc_info(src_alloc);

  unsigned char *src_cuda_ptr =
      (unsigned char *)src_alloc_info.ptr + src.offset;

  cuda_memcpy(dst_cuda_ptr, src_cuda_ptr, size);
}

#else
void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size) {
  TI_NOT_IMPLEMENTED;
}
#endif  // TI_WITH_VULKAN && TI_WITH_CUDA

}  // namespace lang
}  // namespace taichi
