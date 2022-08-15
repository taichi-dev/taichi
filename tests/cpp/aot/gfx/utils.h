#pragma once

#include "taichi/rhi/device.h"

namespace taichi {
namespace lang {
static void write_devalloc(taichi::lang::DeviceAllocation &alloc,
                           const void *data,
                           size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(alloc.device->map(alloc));
  std::memcpy(device_arr_ptr, data, size);
  alloc.device->unmap(alloc);
}

static void load_devalloc(taichi::lang::DeviceAllocation &alloc,
                          void *data,
                          size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(alloc.device->map(alloc));
  std::memcpy(data, device_arr_ptr, size);
  alloc.device->unmap(alloc);
}
}  // namespace lang
}  // namespace taichi
