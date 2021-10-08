#pragma once

#include <cstdint>
#include <vector>

#include "taichi/program/program.h"
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

  void set_item(const std::vector<int> &key, uint64_t val);
  uint64_t get_item(const std::vector<int> &key) const;
  intptr_t get_data_ptr_as_int() const;
  int get_element_size() const;
  int get_nelement() const;

 private:
  Program *program;
  int *data_ptr;
  int nelement;
  int element_size;

  int get_linear_index(std::vector<int> &key) const;
};

}  // namespace lang
}  // namespace taichi
