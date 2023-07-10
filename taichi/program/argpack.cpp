#include "taichi/program/argpack.h"
#include "taichi/program/program.h"
#include "fp16.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

ArgPack::ArgPack(Program *prog, const DataType type) : prog_(prog) {
  auto *old_type = type->get_type()->as<ArgPackType>();
  auto [argpack_type, alloc_size] = prog->get_argpack_type_with_data_layout(
      old_type, prog->get_kernel_argument_data_layout());
  dtype = DataType(argpack_type);
  argpack_alloc_ =
      prog->allocate_memory_on_device(alloc_size, prog->result_buffer);
}

ArgPack::~ArgPack() {
  if (prog_) {
    argpack_alloc_.device->dealloc_memory(argpack_alloc_);
  }
}

intptr_t ArgPack::get_device_allocation_ptr_as_int() const {
  // taichi's own argpack's ptr points to its |DeviceAllocation| on the
  // specified device.
  return reinterpret_cast<intptr_t>(&argpack_alloc_);
}

DeviceAllocation ArgPack::get_device_allocation() const {
  return argpack_alloc_;
}

std::size_t ArgPack::get_nelement() const {
  return dtype->as<ArgPackType>()->elements().size();
}

DataType ArgPack::get_data_type() const {
  return dtype;
}

TypedConstant ArgPack::read(const std::vector<int> &I) const {
  prog_->synchronize();
  size_t offset = dtype->as<ArgPackType>()->get_element_offset(I);
  DataType element_dt = get_element_dt(I);
  size_t size = data_type_size(element_dt);
  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = false;
  alloc_params.host_read = true;
  alloc_params.size = size;
  alloc_params.usage = AllocUsage::Storage;
  auto [staging_buf_, res] =
      this->argpack_alloc_.device->allocate_memory_unique(alloc_params);
  TI_ASSERT(res == RhiResult::success);
  staging_buf_->device->memcpy_internal(
      staging_buf_->get_ptr(), this->argpack_alloc_.get_ptr(offset), size);

  char *device_arr_ptr{nullptr};
  TI_ASSERT(staging_buf_->device->map(
                *staging_buf_, (void **)&device_arr_ptr) == RhiResult::success);

  TypedConstant data(element_dt);
  std::memcpy(&data.value_bits, device_arr_ptr, size);
  staging_buf_->device->unmap(*staging_buf_);

  if (element_dt->is_primitive(PrimitiveTypeID::f16)) {
    float float32 = fp16_ieee_to_fp32_value(data.val_u16);
    data.val_f32 = float32;
  }
  return data;
}

void ArgPack::write(const std::vector<int> &I, TypedConstant val) const {
  size_t offset = dtype->as<ArgPackType>()->get_element_offset(I);
  DataType element_dt = get_element_dt(I);
  size_t size = data_type_size(element_dt);
  if (element_dt->is_primitive(PrimitiveTypeID::f16)) {
    uint16_t float16 = fp16_ieee_from_fp32_value(val.val_f32);
    std::memcpy(&val.value_bits, &float16, 4);
  }

  taichi::lang::Device::AllocParams alloc_params;
  alloc_params.host_write = true;
  alloc_params.host_read = false;
  alloc_params.size = size;
  alloc_params.usage = AllocUsage::Storage;
  auto [staging_buf_, res] =
      this->argpack_alloc_.device->allocate_memory_unique(alloc_params);
  TI_ASSERT(res == RhiResult::success);

  char *device_arr_ptr{nullptr};
  TI_ASSERT(staging_buf_->device->map(
                *staging_buf_, (void **)&device_arr_ptr) == RhiResult::success);

  TI_ASSERT(device_arr_ptr);
  std::memcpy(device_arr_ptr, &val.value_bits, size);

  staging_buf_->device->unmap(*staging_buf_);
  staging_buf_->device->memcpy_internal(this->argpack_alloc_.get_ptr(offset),
                                        staging_buf_->get_ptr(), size);

  prog_->synchronize();
}

void ArgPack::set_arg_int(const std::vector<int> &i, int64 val) const {
  DataType element_dt = dtype->as<ArgPackType>()->get_element_type(i);
  write(i, TypedConstant(element_dt, val));
}

void ArgPack::set_arg_float(const std::vector<int> &i, float64 val) const {
  DataType element_dt = dtype->as<ArgPackType>()->get_element_type(i);
  write(i, TypedConstant(element_dt, val));
}

void ArgPack::set_arg_uint(const std::vector<int> &i, uint64 val) const {
  DataType element_dt = dtype->as<ArgPackType>()->get_element_type(i);
  write(i, TypedConstant(element_dt, val));
}

void ArgPack::set_arg_nested_argpack(int i, const ArgPack &val) const {
  const std::vector<int> indices = {i, TypeFactory::DATA_PTR_POS_IN_ARGPACK};
  DataType element_dt = get_element_dt(indices);
  write(indices,
        TypedConstant(element_dt, val.get_device_allocation_ptr_as_int()));
}

void ArgPack::set_arg_nested_argpack_ptr(int i, intptr_t val) const {
  const std::vector<int> indices = {i, TypeFactory::DATA_PTR_POS_IN_ARGPACK};
  DataType element_dt = get_element_dt(indices);
  write(indices, TypedConstant(element_dt, val));
}

DataType ArgPack::get_element_dt(const std::vector<int> &i) const {
  auto dt = dtype->as<ArgPackType>()->get_element_type(i);
  if (dt->is<PointerType>())
    return PrimitiveType::u64;
  return dt;
}

}  // namespace taichi::lang
