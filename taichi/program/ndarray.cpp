#include <numeric>

#include "taichi/program/ndarray.h"
#include "taichi/program/program.h"
#include "fp16.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

namespace {

size_t flatten_index(const std::vector<int> &shapes,
                     const std::vector<int> &indices) {
  size_t ind = 0;
  for (int i = 0; i < indices.size(); i++) {
    ind = ind * shapes[i] + indices[i];
  }
  return ind;
}
}  // namespace

Ndarray::Ndarray(Program *prog,
                 const DataType type,
                 const std::vector<int> &shape_,
                 ExternalArrayLayout layout_,
                 const DebugInfo &dbg_info_)
    : dtype(type),
      shape(shape_),
      layout(layout_),
      dbg_info(dbg_info_),
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
    ErrorEmitter(
        TaichiIndexWarning(), &dbg_info,
        "Ndarray index might be out of int32 boundary but int64 indexing is "
        "not supported yet.");
  }
  ndarray_alloc_ = prog->allocate_memory_on_device(nelement_ * element_size_,
                                                   prog->result_buffer);
}

Ndarray::Ndarray(DeviceAllocation &devalloc,
                 const DataType type,
                 const std::vector<int> &shape,
                 ExternalArrayLayout layout,
                 const DebugInfo &dbg_info)
    : ndarray_alloc_(devalloc),
      dtype(type),
      shape(shape),
      layout(layout),
      dbg_info(dbg_info),
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
    ErrorEmitter(
        TaichiIndexWarning(), &dbg_info,
        "Ndarray index might be out of int32 boundary but int64 indexing is "
        "not supported yet.");
  }
}

Ndarray::Ndarray(DeviceAllocation &devalloc,
                 const DataType type,
                 const std::vector<int> &shape,
                 const std::vector<int> &element_shape,
                 ExternalArrayLayout layout,
                 const DebugInfo &dbg_info)
    : Ndarray(devalloc,
              TypeFactory::create_tensor_type(element_shape, type),
              shape,
              layout,
              dbg_info) {
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
  alloc_params.usage = AllocUsage::Storage;
  auto [staging_buf_, res] =
      this->ndarray_alloc_.device->allocate_memory_unique(alloc_params);
  TI_ASSERT(res == RhiResult::success);
  staging_buf_->device->memcpy_internal(
      staging_buf_->get_ptr(),
      this->ndarray_alloc_.get_ptr(/*offset=*/index * size), size);

  char *device_arr_ptr{nullptr};
  TI_ASSERT(staging_buf_->device->map(
                *staging_buf_, (void **)&device_arr_ptr) == RhiResult::success);

  TypedConstant data(get_element_data_type());
  std::memcpy(&data.value_bits, device_arr_ptr, size);
  staging_buf_->device->unmap(*staging_buf_);

  if (get_element_data_type()->is_primitive(PrimitiveTypeID::f16)) {
    float float32 = fp16_ieee_to_fp32_value(data.val_u16);
    data.val_f32 = float32;
  }
  return data;
}

void Ndarray::write(const std::vector<int> &I, TypedConstant val) const {
  if (get_element_data_type()->is_primitive(PrimitiveTypeID::f16)) {
    uint16_t float16 = fp16_ieee_from_fp32_value(val.val_f32);
    std::memcpy(&val.value_bits, &float16, 4);
  }

  size_t index = flatten_index(total_shape_, I);
  size_t size_ = data_type_size(get_element_data_type());
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = false;
  alloc_params.size = size_;
  alloc_params.usage = AllocUsage::Storage;
  auto [staging_buf_, res] =
      this->ndarray_alloc_.device->allocate_memory_unique(alloc_params);
  TI_ASSERT(res == RhiResult::success);

  char *device_arr_ptr{nullptr};
  TI_ASSERT(staging_buf_->device->map(
                *staging_buf_, (void **)&device_arr_ptr) == RhiResult::success);

  TI_ASSERT(device_arr_ptr);
  std::memcpy(device_arr_ptr, &val.value_bits, size_);

  staging_buf_->device->unmap(*staging_buf_);
  staging_buf_->device->memcpy_internal(
      this->ndarray_alloc_.get_ptr(index * size_), staging_buf_->get_ptr(),
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
  write(i, TypedConstant(get_element_data_type(), val));
}

void Ndarray::write_float(const std::vector<int> &i, float64 val) {
  write(i, TypedConstant(get_element_data_type(), val));
}

}  // namespace taichi::lang
