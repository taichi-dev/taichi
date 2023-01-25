#include <numeric>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

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
                 ExternalArrayLayout layout_)
    : dtype(type),
      shape(shape_),
      layout(layout_),
      nelement_(std::accumulate(std::begin(shape_),
                                std::end(shape_),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)),
      prog_(prog) {
  // Now that we have two shapes which may be concatenated differently
  // depending on layout, total_shape_ comes handy.
  total_shape_ = shape;
  auto element_shape = data_type_shape(dtype);
  if (layout == ExternalArrayLayout::kAOS) {
    total_shape_.insert(total_shape_.end(), element_shape.begin(),
                        element_shape.end());
  } else if (layout == ExternalArrayLayout::kSOA) {
    total_shape_.insert(total_shape_.begin(), element_shape.begin(),
                        element_shape.end());
  }
  auto total_num_scalar =
      std::accumulate(std::begin(total_shape_), std::end(total_shape_), 1LL,
                      std::multiplies<>());
  if (total_num_scalar > std::numeric_limits<int>::max()) {
    TI_WARN(
        "Ndarray index might be out of int32 boundary but int64 indexing is "
        "not supported yet.");
  }
  ndarray_alloc_ = prog->allocate_memory_ndarray(nelement_ * element_size_,
                                                 prog->result_buffer);
}

Ndarray::Ndarray(DeviceAllocation &devalloc,
                 const DataType type,
                 const std::vector<int> &shape,
                 ExternalArrayLayout layout)
    : ndarray_alloc_(devalloc),
      dtype(type),
      shape(shape),
      layout(layout),
      nelement_(std::accumulate(std::begin(shape),
                                std::end(shape),
                                1,
                                std::multiplies<>())),
      element_size_(data_type_size(dtype)) {
  // When element_shape is specified but layout is not, default layout is AOS.
  auto element_shape = data_type_shape(dtype);
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
  auto total_num_scalar =
      std::accumulate(std::begin(total_shape_), std::end(total_shape_), 1LL,
                      std::multiplies<>());
  if (total_num_scalar > std::numeric_limits<int>::max()) {
    TI_WARN(
        "Ndarray index might be out of int32 boundary but int64 indexing is "
        "not supported yet.");
  }
}

Ndarray::Ndarray(DeviceAllocation &devalloc,
                 const DataType type,
                 const std::vector<int> &shape,
                 const std::vector<int> &element_shape,
                 ExternalArrayLayout layout)
    : Ndarray(devalloc,
              TypeFactory::create_tensor_type(element_shape, type),
              shape,
              layout) {
  TI_ASSERT(type->is<PrimitiveType>());
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

DeviceAllocation Ndarray::get_device_allocation() const {
  return ndarray_alloc_;
}

std::vector<int> Ndarray::get_element_shape() const {
  return data_type_shape(dtype);
}

DataType Ndarray::get_element_data_type() const {
  if (dtype->is<TensorType>()) {
    return dtype->cast<TensorType>()->get_element_type();
  }
  return dtype;
}

std::size_t Ndarray::get_element_size() const {
  return element_size_;
}

std::size_t Ndarray::get_nelement() const {
  return nelement_;
}

TypedConstant Ndarray::read(const std::vector<int> &I) const {
  prog_->synchronize();
  size_t index = flatten_index(total_shape_, I);
  size_t size = data_type_size(get_element_data_type());
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = false;
  alloc_params.host_read = true;
  alloc_params.size = size;
  alloc_params.usage = taichi::lang::AllocUsage::Storage;
  auto staging_buf_ =
      this->ndarray_alloc_.device->allocate_memory_unique(alloc_params);
  staging_buf_->device->memcpy_internal(
      staging_buf_->get_ptr(),
      this->ndarray_alloc_.get_ptr(/*offset=*/index * size), size);

  char *device_arr_ptr{nullptr};
  TI_ASSERT(staging_buf_->device->map(
                *staging_buf_, (void **)&device_arr_ptr) == RhiResult::success);

  TypedConstant data(get_element_data_type());
  std::memcpy(&data.value_bits, device_arr_ptr, size);
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

  T *device_arr_ptr{nullptr};
  TI_ASSERT(staging_buf_->device->map(
                *staging_buf_, (void **)&device_arr_ptr) == RhiResult::success);

  TI_ASSERT(device_arr_ptr);
  device_arr_ptr[0] = val;

  staging_buf_->device->unmap(*staging_buf_);
  staging_buf_->device->memcpy_internal(
      this->ndarray_alloc_.get_ptr(index * sizeof(T)), staging_buf_->get_ptr(),
      size_);

  prog_->synchronize();
}

int64 Ndarray::read_int(const std::vector<int> &i) {
  return read(i).val_int();
}

uint64 Ndarray::read_uint(const std::vector<int> &i) {
  return read(i).val_uint();
}

float64 Ndarray::read_float(const std::vector<int> &i) {
  return read(i).val_float();
}

void Ndarray::write_int(const std::vector<int> &i, int64 val) {
  write<int>(i, val);
}

void Ndarray::write_float(const std::vector<int> &i, float64 val) {
  write<float>(i, val);
}

}  // namespace taichi::lang
