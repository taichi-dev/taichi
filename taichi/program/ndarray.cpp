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
                 const std::vector<int> &shape_,
                 const std::vector<int> &element_shape_,
                 ExternalArrayLayout layout_)
    : dtype(type),
      element_shape(element_shape_),
      shape(shape_),
      layout(layout_),
      nelement_(std::accumulate(std::begin(shape_),
                                std::end(shape_),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype) *
                    std::accumulate(std::begin(element_shape),
                                    std::end(element_shape),
                                    1,
                                    std::multiplies<>())),
      prog_(prog),
      rw_accessors_bank_(&prog->get_ndarray_rw_accessors_bank()) {
  // Now that we have two shapes which may be concatenated differently
  // depending on layout, total_shape_ comes handy.
  total_shape_ = shape;
  if (layout == ExternalArrayLayout::kAOS) {
    total_shape_.insert(total_shape_.end(), element_shape.begin(),
                        element_shape.end());
  } else if (layout == ExternalArrayLayout::kSOA) {
    total_shape_.insert(total_shape_.begin(), element_shape.begin(),
                        element_shape.end());
  }

  ndarray_alloc_ = prog->allocate_memory_ndarray(nelement_ * element_size_,
                                                 prog->result_buffer);
}

Ndarray::Ndarray(DeviceAllocation &devalloc,
                 const DataType type,
                 const std::vector<int> &shape,
                 const std::vector<int> &element_shape,
                 ExternalArrayLayout layout)
    : ndarray_alloc_(devalloc),
      dtype(type),
      element_shape(element_shape),
      shape(shape),
      layout(layout),
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype) *
                    std::accumulate(std::begin(element_shape),
                                    std::end(element_shape),
                                    1,
                                    std::multiplies<>())) {
  // When element_shape is specfied but layout is not, default layout is AOS.
  if (!element_shape.empty() && layout == ExternalArrayLayout::kNull) {
    layout = ExternalArrayLayout::kAOS;
  }
  // Now that we have two shapes which may be concatenated differently
  // depending on layout, total_shape_ comes handy.
  total_shape_ = shape;
  if (layout == ExternalArrayLayout::kAOS) {
    total_shape_.insert(total_shape_.end(), element_shape.begin(),
                        element_shape.end());
  } else if (layout == ExternalArrayLayout::kSOA) {
    total_shape_.insert(total_shape_.begin(), element_shape.begin(),
                        element_shape.end());
  }
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

void set_runtime_ctx_ndarray(RuntimeContext *ctx,
                             int arg_id,
                             Ndarray *ndarray) {
  ctx->set_arg_devalloc(arg_id, ndarray->ndarray_alloc_, ndarray->shape);
}

}  // namespace lang
}  // namespace taichi
