#include <numeric>
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

Ndarray::Ndarray(Program *prog,
                 const DataType type,
                 const std::vector<int> &shape)
    : dtype(type),
      shape(shape),
      num_active_indices(shape.size()),
      program_(prog),
      data_ptr_(nullptr),
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)) {
  prog_ = program_->get_llvm_program_impl();
  data_ptr_ = (uint64_t *)prog_->initialize_llvm_runtime_ndarray(
      nelement_ * element_size_, program_->result_buffer);
}

intptr_t Ndarray::get_data_ptr_as_int() const {
  return reinterpret_cast<intptr_t>(data_ptr_);
}

std::size_t Ndarray::get_element_size() const {
  return element_size_;
}

std::size_t Ndarray::get_nelement() const {
  return nelement_;
}

}  // namespace lang
}  // namespace taichi
