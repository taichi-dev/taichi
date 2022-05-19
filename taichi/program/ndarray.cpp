#include <numeric>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#ifdef TI_WITH_LLVM
#include "taichi/llvm/llvm_context.h"
#include "taichi/llvm/llvm_program.h"
#endif

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
      prog_(prog),
      rw_accessors_bank_(&prog->get_ndarray_rw_accessors_bank()) {
  ndarray_alloc_ = prog->allocate_memory_ndarray(nelement_ * element_size_,
                                                 prog->result_buffer);
}

Ndarray::Ndarray(DeviceAllocation &devalloc,
                 const DataType type,
                 const std::vector<int> &shape)
    : ndarray_alloc_(devalloc),
      dtype(type),
      shape(shape),
      num_active_indices(shape.size()),
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)) {
}

Ndarray::~Ndarray() {
  if (prog_) {
    ndarray_alloc_.device->dealloc_memory(ndarray_alloc_);
  }
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

int64 Ndarray::read_int(const std::vector<int> &i) {
  return rw_accessors_bank_->get(this).read_int(i);
}

uint64 Ndarray::read_uint(const std::vector<int> &i) {
  return rw_accessors_bank_->get(this).read_uint(i);
}

float64 Ndarray::read_float(const std::vector<int> &i) {
  return rw_accessors_bank_->get(this).read_float(i);
}

void Ndarray::write_int(const std::vector<int> &i, int64 val) {
  rw_accessors_bank_->get(this).write_int(i, val);
}

void Ndarray::write_float(const std::vector<int> &i, float64 val) {
  rw_accessors_bank_->get(this).write_float(i, val);
}

void set_runtime_ctx_ndarray(RuntimeContext &ctx,
                             int arg_id,
                             Ndarray &ndarray) {
  ctx.set_arg_devalloc(arg_id, ndarray.ndarray_alloc_, ndarray.shape);
}

}  // namespace lang
}  // namespace taichi
