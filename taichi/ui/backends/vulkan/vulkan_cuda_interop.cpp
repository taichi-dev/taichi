#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/llvm/llvm_context.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

#ifdef _WIN64  // For windows
HANDLE get_device_mem_handle(VkDeviceMemory &mem, VkDevice device) {
  HANDLE handle;

  VkMemoryGetWin32HandleInfoKHR memory_get_win32_handle_info = {};
  memory_get_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  memory_get_win32_handle_info.pNext = NULL;
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
  memory_get_fd_info.pNext = NULL;
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
  CUexternalMemory ext_mem = NULL;
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
  CUexternalMemory ext_mem = NULL;
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
  void *ptr = NULL;
  CUDA_EXTERNAL_MEMORY_BUFFER_DESC desc = {};

  memset(&desc, 0, sizeof(desc));

  desc.offset = offset;
  desc.size = size;

  CUDADriver::get_instance().external_memory_get_mapped_buffer(
      (CUdeviceptr *)&ptr, ext_mem, &desc);
  return ptr;
}

void *get_memory_pointer(VkDeviceMemory mem,
                         VkDeviceSize mem_size,
                         VkDeviceSize offset,
                         VkDeviceSize buffer_size,
                         VkDevice device) {
  auto handle = get_device_mem_handle(mem, device);
  CUexternalMemory externalMem =
      import_vk_memory_object_from_handle(handle, mem_size, false);
  return map_buffer_onto_external_memory(externalMem, offset, buffer_size);
}

// To understand how this works, please read the following resources:
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/vulkanImageCUDA //
// this link uses CUDA runtime api.
// https://stackoverflow.com/questions/55424875/use-vulkan-vkimage-as-a-cuda-cuarray
// // this link contains the necessary information on how to migrate to CUDA
// driver api

CUsurfObject get_image_surface_object_of_external_memory(
    CUexternalMemory external_mem,
    uint64_t offset,
    int width,
    int height,
    int depth) {
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC external_mem_mipmapped_array_desc;

  memset(&external_mem_mipmapped_array_desc, 0,
         sizeof(external_mem_mipmapped_array_desc));

  external_mem_mipmapped_array_desc.offset = offset;
  external_mem_mipmapped_array_desc.numLevels = 1;
  external_mem_mipmapped_array_desc.arrayDesc.Width = width;
  external_mem_mipmapped_array_desc.arrayDesc.Height = height;
  external_mem_mipmapped_array_desc.arrayDesc.Depth = depth;
  external_mem_mipmapped_array_desc.arrayDesc.Format =
      CU_AD_FORMAT_UNSIGNED_INT8;
  external_mem_mipmapped_array_desc.arrayDesc.NumChannels = 4;
  external_mem_mipmapped_array_desc.arrayDesc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

  CUmipmappedArray cuda_mipmapped_image_array;

  CUDADriver::get_instance().external_memory_get_mapped_mipmapped_array(
      &cuda_mipmapped_image_array, external_mem,
      &external_mem_mipmapped_array_desc);

  CUarray cuda_mip_level_array;
  CUDADriver::get_instance().mipmapped_array_get_level(
      &cuda_mip_level_array, cuda_mipmapped_image_array, 0);

  CUDA_RESOURCE_DESC resource_desc;
  memset(&resource_desc, 0, sizeof(resource_desc));
  resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
  resource_desc.res.array.hArray = cuda_mip_level_array;

  CUsurfObject texture_surface_;

  CUDADriver::get_instance().surf_object_create(&texture_surface_,
                                                &resource_desc);
  return texture_surface_;
}

#ifdef _WIN64  // For windows
HANDLE get_vk_semaphore_handle(
    VkExternalSemaphoreHandleTypeFlagBitsKHR external_semaphore_handle_type,
    VkSemaphore &semaphore,
    VkDevice device) {
  HANDLE handle;

  VkSemaphoreGetWin32HandleInfoKHR semaphore_get_win32_handle_info = {};
  semaphore_get_win32_handle_info.sType =
      VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  semaphore_get_win32_handle_info.pNext = NULL;
  semaphore_get_win32_handle_info.semaphore = semaphore;
  semaphore_get_win32_handle_info.handleType = external_semaphore_handle_type;

  auto fpGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
          device, "vkGetSemaphoreWin32HandleKHR");

  if (fpGetSemaphoreWin32HandleKHR == nullptr) {
    TI_ERROR("fpGetSemaphoreWin32HandleKHR is nullptr");
  }

  fpGetSemaphoreWin32HandleKHR(device, &semaphore_get_win32_handle_info,
                               &handle);

  return handle;
}
#else
int get_vk_semaphore_handle(
    VkExternalSemaphoreHandleTypeFlagBitsKHR external_semaphore_handle_type,
    VkSemaphore &semaphore,
    VkDevice device) {
  if (external_semaphore_handle_type ==
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
    int fd;

    VkSemaphoreGetFdInfoKHR semaphore_get_fd_info = {};
    semaphore_get_fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    semaphore_get_fd_info.pNext = NULL;
    semaphore_get_fd_info.semaphore = semaphore;
    semaphore_get_fd_info.handleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;

    auto fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
        device, "vkGetSemaphoreFdKHR");

    if (fpGetSemaphoreFdKHR == nullptr) {
      TI_ERROR("vkGetSemaphoreFdKHR is nullptr");
    }

    fpGetSemaphoreFdKHR(device, &semaphore_get_fd_info, &fd);

    return fd;
  }
  return -1;
}
#endif

void cuda_vk_semaphore_signal(CUexternalSemaphore ext_smaphore,
                              CUstream stream) {
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS ext_smaphore_signal_params;
  memset(&ext_smaphore_signal_params, 0, sizeof(ext_smaphore_signal_params));

  ext_smaphore_signal_params.params.fence.value = 0;
  ext_smaphore_signal_params.flags = 0;
  CUDADriver::get_instance().signal_external_semaphore_async(
      &ext_smaphore, &ext_smaphore_signal_params, 1, stream);
}

void cuda_vk_semaphore_wait(CUexternalSemaphore ext_smaphore, CUstream stream) {
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS ext_smaphore_wait_params;

  memset(&ext_smaphore_wait_params, 0, sizeof(ext_smaphore_wait_params));

  ext_smaphore_wait_params.params.fence.value = 0;
  ext_smaphore_wait_params.flags = 0;

  CUDADriver::get_instance().wait_external_semaphore_async(
      &ext_smaphore, &ext_smaphore_wait_params, 1, stream);
}

CUexternalSemaphore cuda_vk_import_semaphore(VkSemaphore semaphore,
                                             VkDevice device) {
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC external_semaphore_handle_desc;
  memset(&external_semaphore_handle_desc, 0,
         sizeof(external_semaphore_handle_desc));
#ifdef _WIN64
  external_semaphore_handle_desc.type =
      CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
  external_semaphore_handle_desc.handle.win32.handle = get_vk_semaphore_handle(
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, semaphore, device);
#else
  external_semaphore_handle_desc.type =
      CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
  external_semaphore_handle_desc.handle.fd = get_vk_semaphore_handle(
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, semaphore, device);
#endif
  external_semaphore_handle_desc.flags = 0;

  CUexternalSemaphore result;

  CUDADriver::get_instance().import_external_semaphore(
      &result, &external_semaphore_handle_desc);
  return result;
}

void cuda_memcpy(void *dst, void *src, size_t size) {
  CUDADriver::get_instance().memcpy_device_to_device(dst, src, size);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
