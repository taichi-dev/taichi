#pragma once

#include "taichi/ui/utils/utils.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/jit/jit_session.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/backends/cuda/jit_cuda.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

#ifdef _WIN64  // For windows
HANDLE get_device_mem_handle(VkDeviceMemory &mem, VkDevice device);
CUexternalMemory import_vk_memory_object_from_handle(HANDLE handle,
                                                     unsigned long long size,
                                                     bool is_dedicated);
HANDLE get_vk_semaphore_handle(
    VkExternalSemaphoreHandleTypeFlagBitsKHR external_semaphore_handle_type,
    VkSemaphore &semaphore,
    VkDevice device);
#else
int get_device_mem_handle(VkDeviceMemory &mem, VkDevice device);
CUexternalMemory import_vk_memory_object_from_handle(int fd,
                                                     unsigned long long size,
                                                     bool is_dedicated);
int get_vk_semaphore_handle(
    VkExternalSemaphoreHandleTypeFlagBitsKHR external_semaphore_handle_type,
    VkSemaphore &semaphore,
    VkDevice device);
#endif

void *map_buffer_onto_external_memory(CUexternalMemory ext_mem,
                                      unsigned long long offset,
                                      unsigned long long size);

void *get_memory_pointer(VkDeviceMemory mem,
                         VkDeviceSize mem_size,
                         VkDeviceSize offset,
                         VkDeviceSize buffer_size,
                         VkDevice device);

CUsurfObject get_image_surface_object_of_external_memory(
    CUexternalMemory external_mem,
    uint64_t offset,
    int width,
    int height,
    int depth);

CUexternalSemaphore cuda_vk_import_semaphore(VkSemaphore semaphore,
                                             VkDevice device);

void cuda_vk_semaphore_signal(CUexternalSemaphore ext_smaphore,
                              CUstream stream = 0);

void cuda_vk_semaphore_wait(CUexternalSemaphore ext_smaphore,
                            CUstream stream = 0);

class InteropCUDALauncher {
 public:
  static InteropCUDALauncher &instance() {
    static InteropCUDALauncher instance;
    return instance;
  }

 public:
  InteropCUDALauncher(InteropCUDALauncher const &) = delete;
  void operator=(InteropCUDALauncher const &) = delete;
  taichi::lang::JITSessionCUDA *session();
  taichi::lang::JITModuleCUDA *module();

  void update_renderables_vertices(float *vbo,
                                   int stride,
                                   float *data,
                                   int num_vertices,
                                   int num_components,
                                   int offset_bytes);

  void update_renderables_indices(int *ibo, int *indices, int num_indices);

  template <typename T>
  void copy_to_texture_buffer(T *src,
                              unsigned char *dest,
                              int width,
                              int height,
                              int actual_width,
                              int actual_height,
                              int channels);

 private:
  InteropCUDALauncher();

  std::unique_ptr<taichi::lang::JITSessionCUDA> session_{nullptr};
  taichi::lang::JITModuleCUDA *module_{nullptr};
};

void update_renderables_vertices_x64(float *vbo,
                                     int stride,
                                     float *data,
                                     int num_vertices,
                                     int num_components,
                                     int offset_bytes);

void update_renderables_indices_x64(int *ibo, int *indices, int num_indices);

template <typename T>
void copy_to_texture_buffer_x64(T *src,
                                unsigned char *dest,
                                int width,
                                int height,
                                int actual_width,
                                int actual_height,
                                int channels);

}  // namespace vulkan

TI_UI_NAMESPACE_END
