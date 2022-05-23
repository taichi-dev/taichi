#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/backends/device.h"

namespace taichi {
namespace lang {

class Program;
class LlvmProgramImpl;
class NdarrayRwAccessorsBank;

class Ndarray {
 public:
  /* Constructs a Ndarray managed by Program.
   * Memory allocation and deallocation is handled by Program.
   * TODO: Ideally Ndarray shouldn't worry about memory alloc/dealloc at all.
   */
  explicit Ndarray(Program *prog,
                   const DataType type,
                   const std::vector<int> &shape);

  /* Constructs a Ndarray from an existing DeviceAllocation
   * It doesn't handle the allocation and deallocation.
   */
  explicit Ndarray(DeviceAllocation &devalloc,
                   const DataType type,
                   const std::vector<int> &shape);
  DeviceAllocation ndarray_alloc_{kDeviceNullAllocation};
  DataType dtype;
  // Invariant: Since ndarray indices are flattened for vector/matrix, this is
  // always true:
  //   num_active_indices = shape.size()
  std::vector<int> shape;
  int num_active_indices{0};

  intptr_t get_data_ptr_as_int() const;
  intptr_t get_device_allocation_ptr_as_int() const;
  std::size_t get_element_size() const;
  std::size_t get_nelement() const;
  int64 read_int(const std::vector<int> &i);
  uint64 read_uint(const std::vector<int> &i);
  float64 read_float(const std::vector<int> &i);
  void write_int(const std::vector<int> &i, int64 val);
  void write_float(const std::vector<int> &i, float64 val);
  ~Ndarray();

 private:
  std::size_t nelement_{1};
  std::size_t element_size_{1};

  Program *prog_{nullptr};
  // TODO: maybe remove these?
  NdarrayRwAccessorsBank *rw_accessors_bank_{nullptr};
};

// TODO: move this as a method inside RuntimeContext once Ndarray is decoupled
// with Program
void set_runtime_ctx_ndarray(RuntimeContext *ctx, int arg_id, Ndarray *ndarray);
}  // namespace lang
}  // namespace taichi
