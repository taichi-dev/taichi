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
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)),
      prog_impl_(prog->get_llvm_program_impl()) {
#ifdef TI_WITH_LLVM
  ndarray_alloc_ = prog_impl_->allocate_memory_ndarray(
      nelement_ * element_size_, prog->result_buffer);

  data_ptr_ = prog_impl_->get_ndarray_alloc_info_ptr(ndarray_alloc_);
#else
  TI_ERROR("Llvm disabled");
#endif
}

void Ndarray::release() {
  prog_impl_->release_memory_ndarray(ndarray_alloc_);
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
