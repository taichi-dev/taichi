#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/backends/device.h"

#ifdef TI_WITH_LLVM
#include "taichi/llvm/llvm_context.h"
#include "taichi/llvm/llvm_program.h"
#endif

namespace taichi {
namespace lang {

class Program;

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
  ~Ndarray();

 private:
  DeviceAllocation ndarray_alloc_{kDeviceNullAllocation};
  // Invariant:
  //   data_ptr_ is not nullptr iff arch is a llvm backend
  uint64_t *data_ptr_{nullptr};
  std::size_t nelement_{1};
  std::size_t element_size_{1};
  // Ndarrays manage their own |DeviceAllocation| so this must be shared with
  // |OpenGlRuntime|. Without the ownership, when the program exits |device_|
  // might be destructed earlier than Ndarray object, leaving a segfault when
  // you try to deallocate in Ndarray destructor.
  // Note that we might consider changing this logic later if we implement
  // dynamic tensor rematerialization.
  std::shared_ptr<Device> device_{nullptr};
};

}  // namespace lang
}  // namespace taichi
