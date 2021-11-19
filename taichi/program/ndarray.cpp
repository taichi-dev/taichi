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
      element_size_(data_type_size(dtype)) {
#ifdef TI_WITH_LLVM
  ndarray_alloc_ = prog->allocate_memory_ndarray(nelement_ * element_size_,
                                                 prog->result_buffer);

  if (arch_is_cpu(prog->config.arch) || prog->config.arch == Arch::cuda) {
    // Keep this information for TNG.
    data_ptr_ = prog->get_llvm_program_impl()->get_ndarray_alloc_info_ptr(
        ndarray_alloc_);
  }

  // taichi's own ndarray's ptr points to its |DeviceAllocation| on the
  // specified device. Note that torch-based ndarray's ptr is a raw ptr but
  // we'll get rid of it soon.
  device_allocation_ptr_ = (uint64_t *)&ndarray_alloc_;
#else
  TI_ERROR("Llvm disabled");
#endif
}

intptr_t Ndarray::get_data_ptr_as_int() const {
  return reinterpret_cast<intptr_t>(data_ptr_);
}

intptr_t Ndarray::get_device_allocation_ptr_as_int() const {
  return reinterpret_cast<intptr_t>(device_allocation_ptr_);
}

std::size_t Ndarray::get_element_size() const {
  return element_size_;
}

std::size_t Ndarray::get_nelement() const {
  return nelement_;
}

}  // namespace lang
}  // namespace taichi
