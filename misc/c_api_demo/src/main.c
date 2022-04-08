#include <stdio.h>
#include <stdlib.h>

#include "c_api/include/taichi/aot/aot.h"
#include "c_api/include/taichi/backends/vulkan_device.h"

int main() {
  // 1.2, copied from VK_API_VERSION_1_2
  const uint32_t kVulkanApiVersion = 4202496;
  Taichi_EmbeddedVulkanDevice *evd =
      taichi_make_embedded_vulkan_device(kVulkanApiVersion, NULL, 0, NULL, 0);
  Taichi_VulkanDevice *vk_dev = taichi_get_vulkan_device(evd);
  uint64_t *host_result_buffer = malloc(sizeof(uint64_t) * 4096);
  Taichi_VulkanRuntime *vk_rtm =
      taichi_make_vulkan_runtime(host_result_buffer, vk_dev);
  Taichi_AotModule *m =
      taichi_make_vulkan_aot_module("../src/generated/", vk_rtm);
  taichi_vulkan_add_root_buffer(vk_rtm,
                                taichi_get_root_size_from_aot_module(m));

  printf("AotModule m=%lld\n", (uint64_t)m);
  Taichi_Kernel *fill_k = taichi_get_kernel_from_aot_module(m, "fill");
  printf("fill_k=%lld\n", (uint64_t)fill_k);

  Taichi_NdShape *x_shape = malloc(sizeof(Taichi_NdShape) + sizeof(int32_t));
  x_shape->length = 2;
  x_shape->data[0] = 2;
  x_shape->data[1] = 8;
  const int kXShapeLinear = x_shape->data[0] * x_shape->data[1];

  Taichi_DeviceAllocParams x_alloc_params;
  x_alloc_params.size = sizeof(uint32_t) * kXShapeLinear;
  x_alloc_params.host_read = true;
  x_alloc_params.host_write = false;
  x_alloc_params.export_sharing = false;
  Taichi_DeviceAllocation *x_ndarray =
      taichi_allocate_device_memory(vk_dev, &x_alloc_params);
  Taichi_RuntimeContext *ctx = taichi_make_runtime_context();

  taichi_set_runtime_context_arg_scalar_ndarray(ctx, /*param_i=*/0, x_ndarray,
                                                x_shape);
  taichi_set_runtime_context_arg_i32(ctx, /*param_i=*/1, /*val=*/100);
  taichi_launch_kernel(fill_k, ctx);
  printf("launched fill kernel\n");
  taichi_vulkan_synchronize(vk_rtm);
  printf("Vulkan synchronized\n");

  int32_t *data = taichi_map_device_allocation(vk_dev, x_ndarray);
  for (int i = 0; i < kXShapeLinear; ++i) {
    printf("x[%d]=%d\n", i, data[i]);
  }
  taichi_unmap_device_allocation(vk_dev, x_ndarray);

  taichi_deallocate_device_memory(vk_dev, x_ndarray);
  free(x_shape);
  taichi_destroy_runtime_context(ctx);
  taichi_destroy_vulkan_aot_module(m);
  taichi_destroy_vulkan_runtime(vk_rtm);
  free(host_result_buffer);
  taichi_destroy_embedded_vulkan_device(evd);
  return 0;
}
