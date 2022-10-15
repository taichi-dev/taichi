#pragma once

#include "taichi/rhi/device.h"

namespace taichi::lang {

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

void memcpy_vulkan_to_cuda(DevicePtr dst, DevicePtr src, uint64_t size);

}  // namespace taichi::lang
