#pragma once

#include "taichi/backends/device.h"

namespace taichi {
namespace lang {

void memcpy_cpu_to_vulkan_via_staging(DevicePtr dst,
                                      DevicePtr staging,
                                      DevicePtr src,
                                      uint64_t size);

}  // namespace lang
}  // namespace taichi
