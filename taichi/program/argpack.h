#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class Program;

class TI_DLL_EXPORT ArgPack {
 public:
  /* Constructs a ArgPack managed by Program.
   * Memory allocation and deallocation is handled by Program.
   */
  explicit ArgPack(Program *prog,
                   const DataType type);

  DeviceAllocation argpack_alloc_{kDeviceNullAllocation};
  DataType dtype;

  DataType get_data_type() const;
  intptr_t get_device_allocation_ptr_as_int() const;
  DeviceAllocation get_device_allocation() const;
  std::size_t get_nelement() const;

  ~ArgPack();

 private:
  Program *prog_{nullptr};
};
}