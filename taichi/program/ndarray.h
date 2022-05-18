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
  explicit Ndarray(Program *prog,
                   const DataType type,
                   const std::vector<int> &shape);

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
  void fill_float(float val);
  void fill_int(int32_t val);
  void fill_uint(uint32_t val);
  int64 read_int(const std::vector<int> &i);
  uint64 read_uint(const std::vector<int> &i);
  float64 read_float(const std::vector<int> &i);
  void write_int(const std::vector<int> &i, int64 val);
  void write_float(const std::vector<int> &i, float64 val);
  ~Ndarray();

 private:
  void buffer_fill(uint32_t val);

  DeviceAllocation ndarray_alloc_{kDeviceNullAllocation};
  // Invariant:
  //   data_ptr_ is not nullptr iff arch is a llvm backend
  uint64_t *data_ptr_{nullptr};
  std::size_t nelement_{1};
  std::size_t element_size_{1};

  Program *prog_{nullptr};
  // TODO: maybe remove these?
  LlvmProgramImpl *prog_impl_{nullptr};
  NdarrayRwAccessorsBank *rw_accessors_bank_{nullptr};
};

}  // namespace lang
}  // namespace taichi
