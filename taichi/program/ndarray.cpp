#include <numeric>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi {
namespace lang {

namespace {

size_t flatten_index(const std::vector<int> &shapes,
                     const std::vector<int> &indices) {
  TI_ASSERT(shapes.size() == indices.size());
  if (indices.size() == 1) {
    return indices[0];
  } else {
    size_t ind = indices[0];
    for (int i = 1; i < indices.size(); i++) {
      ind = ind * shapes[i] + indices[i];
    }
    return ind;
  }
}
}  // namespace
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
      prog_(prog) {
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
  // When element_shape is specified but layout is not, default layout is AOS.
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
    // prog_->flush();
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

template <typename T>
T Ndarray::read(const std::vector<int> &I) const {
  prog_->synchronize();
  size_t index = flatten_index(total_shape_, I);
  size_t size_ = sizeof(T);
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = false;
  alloc_params.host_read = true;
  alloc_params.size = size_;
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  auto staging_buf_ =
      this->ndarray_alloc_.device->allocate_memory_unique(alloc_params);
  staging_buf_->device->memcpy_internal(
      staging_buf_->get_ptr(),
      this->ndarray_alloc_.get_ptr(/*offset=*/index * sizeof(T)), size_);

  char *const device_arr_ptr =
      reinterpret_cast<char *>(staging_buf_->device->map(*staging_buf_));
  TI_ASSERT(device_arr_ptr);

  T data;
  std::memcpy(&data, device_arr_ptr, size_);
  staging_buf_->device->unmap(*staging_buf_);
  return data;
}

template <typename T>
void Ndarray::write(const std::vector<int> &I, T val) const {
  size_t index = flatten_index(total_shape_, I);
  size_t size_ = sizeof(T);
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = false;
  alloc_params.size = size_;
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  auto staging_buf_ =
      this->ndarray_alloc_.device->allocate_memory_unique(alloc_params);

  T *const device_arr_ptr =
      reinterpret_cast<T *>(staging_buf_->device->map(*staging_buf_));

  TI_ASSERT(device_arr_ptr);
  device_arr_ptr[0] = val;

  staging_buf_->device->unmap(*staging_buf_);
  staging_buf_->device->memcpy_internal(
      this->ndarray_alloc_.get_ptr(index * sizeof(T)), staging_buf_->get_ptr(),
      size_);

  prog_->synchronize();
}

int64 Ndarray::read_int(const std::vector<int> &i) {
  return read<int>(i);
}

uint64 Ndarray::read_uint(const std::vector<int> &i) {
  return read<uint>(i);
}

float64 Ndarray::read_float(const std::vector<int> &i) {
  return read<float>(i);
}

void Ndarray::write_int(const std::vector<int> &i, int64 val) {
  write<int>(i, val);
}

void Ndarray::write_float(const std::vector<int> &i, float64 val) {
  write<float>(i, val);
}

}  // namespace lang
}  // namespace taichi
