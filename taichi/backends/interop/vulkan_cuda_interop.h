#pragma once

#include "taichi/backends/device.h"

namespace taichi {
namespace lang {

void memcpy_cuda_to_vulkan(DevicePtr dst, DevicePtr src, uint64_t size);

}  // namespace lang
}  // namespace taichi
