#include <numeric>
#include "taichi/program/ndarray.h"

namespace taichi {
namespace lang {

Ndarray::Ndarray(Program *prog,
                 const DataType type,
                 const std::vector<int> &shape)
    : dtype(type),
      shape(shape),
      program_(prog),
      data_ptr_(nullptr),
      nelement_(std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>())),
      element_size_(data_type_size(dtype)) {
  prog_ = program_->get_llvm_program_impl();
  data_ptr_ = (uint64_t*)prog_->initialize_llvm_runtime_ndarray(nelement_ * element_size_, program_->result_buffer);
}

void Ndarray::set_item(const std::vector<int> &key, uint64_t val) {
  int pos = get_linear_index(key);
  data_ptr_[pos] = val;
}

uint64_t Ndarray::get_item(const std::vector<int> &key) const {
  int pos = get_linear_index(key);
  return data_ptr_[pos];
}

intptr_t Ndarray::get_data_ptr_as_int() const {
  return reinterpret_cast<std::intptr_t>(data_ptr_);
}

std::size_t Ndarray::get_element_size() const {
  return element_size_;
}

std::size_t Ndarray::get_nelement() const {
  return nelement_;
}

int Ndarray::get_linear_index(const std::vector<int> &key) const {
  assert(key.size() == shape.size());

  // TODO, generalize this to ND?
  int ret{0};
  if (shape.size() == 1) {
    ret = key[0];
  } else if (shape.size() == 2) {
    ret = key[0] * shape[1] + key[1];
  } else {
    assert(0);
  }
  return ret;
}
}  // namespace lang
}  // namespace taichi
