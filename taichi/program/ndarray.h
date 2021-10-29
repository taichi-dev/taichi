#pragma once

#include <cstdint>
#include <vector>

#include "taichi/inc/constants.h"
#include "taichi/ir/type_utils.h"
#include "taichi/llvm/llvm_context.h"
#include "taichi/llvm/llvm_program.h"

namespace taichi {
namespace lang {

class Program;

class Ndarray {
 public:
  explicit Ndarray(Program *prog,
                   const DataType type,
                   const std::vector<int> &shape);

  DataType dtype;
  std::vector<int> shape;
  int num_active_indices{0};

  intptr_t get_data_ptr_as_int() const;
  std::size_t get_element_size() const;
  std::size_t get_nelement() const;

 private:
  uint64_t *data_ptr_{nullptr};
  std::size_t nelement_{1};
  std::size_t element_size_{1};
};

}  // namespace lang
}  // namespace taichi
