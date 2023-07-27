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
  explicit ArgPack(Program *prog, const DataType type);

  DeviceAllocation argpack_alloc_{kDeviceNullAllocation};
  DataType dtype;

  DataType get_data_type() const;
  intptr_t get_device_allocation_ptr_as_int() const;
  DeviceAllocation get_device_allocation() const;
  std::size_t get_nelement() const;

  TypedConstant read(const std::vector<int> &I) const;
  void write(const std::vector<int> &I, TypedConstant val) const;
  void set_arg_int(const std::vector<int> &i, int64 val) const;
  void set_arg_uint(const std::vector<int> &i, uint64 val) const;
  void set_arg_float(const std::vector<int> &i, float64 val) const;
  void set_arg_nested_argpack(int i, const ArgPack &val) const;
  void set_arg_nested_argpack_ptr(int i, intptr_t val) const;

  ~ArgPack();

 private:
  Program *prog_{nullptr};

  DataType get_element_dt(const std::vector<int> &i) const;
};
}  // namespace taichi::lang
