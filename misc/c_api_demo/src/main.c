#include <stdio.h>
#include <stdlib.h>

#include "c_api/include/taichi/aot.h"

int main() {
  // 1.2, copied from VK_API_VERSION_1_2
  const uint32_t kVulkanApiVersion = 4202496;
  EmbeddedVulkanDevice* evd =
      make_embedded_vulkan_device(kVulkanApiVersion, NULL, 0, NULL, 0);
  VulkanDevice* vk_dev = get_vulkan_device(evd);
  uint64_t* host_result_buffer = malloc(sizeof(uint64_t) * 4096);
  VulkanRuntime* vk_rtm = make_vulkan_runtime(host_result_buffer, vk_dev);
  AotModule* m = make_vulkan_aot_module("../src/generated/", vk_rtm);
  vulkan_add_root_buffer(vk_rtm, get_root_size_from_aot_module(m));

  printf("AotModule m=%lld\n", (uint64_t)m);
  TaichiKernel* fill_k = get_taichi_kernel(m, "fill");
  printf("fill_k=%lld\n", (uint64_t)fill_k);

  NdShape* x_shape = malloc(sizeof(NdShape) + sizeof(int32_t));
  x_shape->length = 2;
  x_shape->data[0] = 2;
  x_shape->data[1] = 8;
  const int kXShapeLinear = x_shape->data[0] * x_shape->data[1];

  DeviceAllocParams x_alloc_params;
  x_alloc_params.size = sizeof(uint32_t) * kXShapeLinear;
  x_alloc_params.host_read = true;
  x_alloc_params.host_write = false;
  x_alloc_params.export_sharing = false;
  DeviceAllocation* x_ndarray = vulkan_allocate_memory(vk_dev, &x_alloc_params);
  TaichiRuntimeContext* ctx = make_runtime_context();

  set_runtime_context_arg_scalar_ndarray(ctx, /*param_i=*/0, x_ndarray,
                                         x_shape);
  set_runtime_context_arg_i32(ctx, /*param_i=*/1, /*val=*/100);
  launch_taichi_kernel(fill_k, ctx);
  printf("launched fill kernel\n");
  vulkan_synchronize(vk_rtm);
  printf("Vulkan synchronized\n");

  int32_t* data = vulkan_map_memory(vk_dev, x_ndarray);
  for (int i = 0; i < kXShapeLinear; ++i) {
    printf("x[%d]=%d\n", i, data[i]);
  }
  vulkan_unmap_memory(vk_dev, x_ndarray);

  vulkan_dealloc_memory(vk_dev, x_ndarray);
  free(x_shape);
  destroy_runtime_context(ctx);
  destroy_vulkan_aot_module(m);
  destroy_vulkan_runtime(vk_rtm);
  free(host_result_buffer);
  destroy_embedded_vulkan_device(evd);
  return 0;
}
