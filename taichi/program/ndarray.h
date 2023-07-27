#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/type_utils.h"
#include "taichi/rhi/device.h"

namespace taichi::lang {

class Program;
class NdarrayRwAccessorsBank;

class TI_DLL_EXPORT Ndarray {
 public:
  /* Constructs a Ndarray managed by Program.
   * Memory allocation and deallocation is handled by Program.
   * TODO: Ideally Ndarray shouldn't worry about memory alloc/dealloc at all.
   */
  explicit Ndarray(Program *prog,
                   const DataType type,
                   const std::vector<int> &shape,
                   ExternalArrayLayout layout = ExternalArrayLayout::kNull,
                   const DebugInfo &dbg_info = DebugInfo());

  /* Constructs a Ndarray from an existing DeviceAllocation.
   * It doesn't handle the allocation and deallocation.
   * You can see a Ndarray as a view or interpretation of DeviceAllocation
   * with specified dtype & layout.
   */
  explicit Ndarray(DeviceAllocation &devalloc,
                   const DataType type,
                   const std::vector<int> &shape,
                   ExternalArrayLayout layout = ExternalArrayLayout::kNull,
                   const DebugInfo &dbg_info = DebugInfo());

  /* Constructs a Ndarray from an existing DeviceAllocation.
   * This is an overloaded constructor for constructing Ndarray with TensorType
   * elements "type" is expected to be PrimitiveType
   */
  explicit Ndarray(DeviceAllocation &devalloc,
                   const DataType type,
                   const std::vector<int> &shape,
                   const std::vector<int> &element_shape,
                   ExternalArrayLayout layout = ExternalArrayLayout::kNull,
                   const DebugInfo &dbg_info = DebugInfo());

  DeviceAllocation ndarray_alloc_{kDeviceNullAllocation};
  DataType dtype;
  // Invariant: Since ndarray indices are flattened for vector/matrix, this is
  // always true:
  //   num_active_indices = shape.size()
  std::vector<int> shape;
  ExternalArrayLayout layout{ExternalArrayLayout::kNull};
  DebugInfo dbg_info;

  std::vector<int> get_element_shape() const;
  DataType get_element_data_type() const;
  intptr_t get_data_ptr_as_int() const;
  intptr_t get_device_allocation_ptr_as_int() const;
  DeviceAllocation get_device_allocation() const;
  std::size_t get_element_size() const;
  std::size_t get_nelement() const;
  TypedConstant read(const std::vector<int> &I) const;
  void write(const std::vector<int> &I, TypedConstant val) const;
  int64 read_int(const std::vector<int> &i);
  uint64 read_uint(const std::vector<int> &i);
  float64 read_float(const std::vector<int> &i);
  void write_int(const std::vector<int> &i, int64 val);
  void write_float(const std::vector<int> &i, float64 val);

  const std::vector<int> &total_shape() const {
    return total_shape_;
  }
  ~Ndarray();

 private:
  std::size_t nelement_{1};
  std::size_t element_size_{1};
  std::vector<int> total_shape_;

  Program *prog_{nullptr};
};

}  // namespace taichi::lang
