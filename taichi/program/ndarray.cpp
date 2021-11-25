#include <numeric>
#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

Ndarray::Ndarray(Program *prog,
                 const DataType type,
                 const std::vector<int> &shape)
    : prog_(prog),
      dtype(type),
      shape(shape),
      num_active_indices(shape.size()),
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)),
      device_(prog_->get_device_shared()) {
  ndarray_alloc_ = prog_->allocate_memory_ndarray(nelement_ * element_size_,
                                                 prog_->result_buffer);
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(prog_->config.arch) || prog_->config.arch == Arch::cuda) {
    // For the LLVM backends, device allocation is a physical pointer.
    data_ptr_ = prog_->get_llvm_program_impl()->get_ndarray_alloc_info_ptr(
        ndarray_alloc_);
  }
#else
  TI_ERROR("Llvm disabled");
#endif
}

Ndarray::~Ndarray() {
  if (device_) {
#ifdef TI_WITH_OPENGL
    device_->dealloc_memory(ndarray_alloc_);
#elif TI_WITH_LLVM
    // cpu and cuda backend use the preallocated memory from the runtime module
    if (arch_is_cpu(prog_->config.arch) || prog_->config.arch == Arch::cuda) {
      device_->dealloc_memory_runtime(ndarray_alloc_);
    }
#else
    TI_ERROR("Arch is not supported by Ndarray");
#endif
  }
}

intptr_t Ndarray::get_data_ptr_as_int() const {
  return reinterpret_cast<intptr_t>(data_ptr_);
}

intptr_t Ndarray::get_device_allocation_ptr_as_int() const {
  // taichi's own ndarray's ptr points to its |DeviceAllocation| on the
  // specified device. Note that torch-based ndarray's ptr is a raw ptr but
  // we'll get rid of it soon.
  return reinterpret_cast<intptr_t>(&ndarray_alloc_);
}

std::size_t Ndarray::get_element_size() const {
  return element_size_;
}

std::size_t Ndarray::get_nelement() const {
  return nelement_;
}

}  // namespace lang
}  // namespace taichi
