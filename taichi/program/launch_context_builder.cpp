#include <taichi/program/launch_context_builder.h>
#define TI_RUNTIME_HOST
#include <taichi/program/context.h>
#undef TI_RUNTIME_HOST
#include "fp16.h"

namespace taichi::lang {

namespace {
template <typename T>
inline std::vector<T> concatenate_vector(const std::vector<T> &lhs,
                                         const std::vector<T> &rhs) {
  std::vector<T> result;
  result.assign(lhs.begin(), lhs.end());
  result.insert(result.end(), rhs.begin(), rhs.end());
  return result;
}
}  // namespace

LaunchContextBuilder::LaunchContextBuilder(CallableBase *kernel)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<RuntimeContext>()),
      ctx_(owned_ctx_.get()),
      arg_buffer_(std::make_unique<char[]>(kernel->args_size)),
      result_buffer_(std::make_unique<char[]>(kernel->ret_size)),
      ret_type_(kernel->ret_type),
      arg_buffer_size(kernel->args_size),
      args_type(kernel->args_type),
      result_buffer_size(kernel->ret_size) {
  ctx_->result_buffer = (uint64 *)result_buffer_.get();
  ctx_->arg_buffer = arg_buffer_.get();
}

void LaunchContextBuilder::set_arg_float(const std::vector<int> &arg_id,
                                         float64 d) {
  auto dt = kernel_->args_type->get_element_type(arg_id);
  TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  PrimitiveTypeID typeId = dt->as<PrimitiveType>()->type;

  switch (typeId) {
#define PER_C_TYPE(tp, ctype)  \
  case PrimitiveTypeID::tp:    \
    set_arg(arg_id, (ctype)d); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      uint16 half = fp16_ieee_from_fp32_value((float32)d);
      set_arg(arg_id, half);
      break;
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

template <typename T>
void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices, T d) {
  auto dt = kernel_->args_type->get_element_type(arg_indices);

  TI_ASSERT(dt->is<PrimitiveType>() || dt->is<PointerType>());
  if (dt->is<PointerType>()) {
    set_struct_arg_impl(arg_indices, (uint64)d);
    return;
  }
  PrimitiveTypeID typeId = dt->as<PrimitiveType>()->type;

  switch (typeId) {
#define PER_C_TYPE(tp, ctype)                   \
  case PrimitiveTypeID::tp:                     \
    set_struct_arg_impl(arg_indices, (ctype)d); \
    break;
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      uint16 half = fp16_ieee_from_fp32_value((float32)d);
      set_struct_arg_impl(arg_indices, half);
      break;
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_ndarray_ptrs(const std::vector<int> &arg_id,
                                            uint64 data_ptr,
                                            uint64 grad_ptr) {
  auto param_indices = arg_id;
  param_indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
  set_struct_arg(param_indices, data_ptr);
  if (kernel_->nested_parameters[arg_id].needs_grad) {
    param_indices.back() = TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
    set_struct_arg(param_indices, grad_ptr);
  }
}

void LaunchContextBuilder::set_argpack_ptr(const std::vector<int> &arg_id,
                                           uint64 data_ptr) {
  auto param_indices = arg_id;
  param_indices.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
  set_struct_arg(param_indices, data_ptr);
}

template void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices,
                                                   uint64 v);
template void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices,
                                                   int64 v);
template void LaunchContextBuilder::set_struct_arg(std::vector<int> arg_indices,
                                                   float64 v);

void LaunchContextBuilder::set_arg_int(const std::vector<int> &arg_id,
                                       int64 d) {
  auto dt = kernel_->args_type->get_element_type(arg_id);

  TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    set_arg(arg_id, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    set_arg(arg_id, (uint1)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    set_arg(arg_id, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    set_arg(arg_id, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    set_arg(arg_id, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    set_arg(arg_id, (uint64)d);
  } else {
    TI_INFO(dt->to_string());
    TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_arg_uint(const std::vector<int> &arg_id,
                                        uint64 d) {
  set_arg_int(arg_id, d);
}

template <>
void LaunchContextBuilder::set_arg<TypedConstant>(const std::vector<int> &i,
                                                  TypedConstant d) {
  if (is_real(d.dt)) {
    set_arg_float(i, d.val_float());
  } else {
    if (is_signed(d.dt)) {
      set_arg_int(i, d.val_int());
    } else {
      set_arg_uint(i, d.val_uint());
    }
  }
}

template <typename T>
void LaunchContextBuilder::set_struct_arg_impl(std::vector<int> arg_indices,
                                               T v) {
  int offset = args_type->get_element_offset(arg_indices);
  TI_ASSERT(offset + sizeof(T) <= arg_buffer_size);
  *(T *)(ctx_->arg_buffer + offset) = v;
}

template <typename T>
T LaunchContextBuilder::get_arg(const std::vector<int> &i) {
  return get_struct_arg<T>(i);
}

template <typename T>
T LaunchContextBuilder::get_struct_arg(std::vector<int> arg_indices) {
  int offset = args_type->get_element_offset(arg_indices);
  TI_ASSERT(offset + sizeof(T) <= arg_buffer_size);
  return *(T *)(ctx_->arg_buffer + offset);
}

template <typename T>
void LaunchContextBuilder::set_arg(const std::vector<int> &i, T v) {
  set_struct_arg_impl(i, v);
  set_array_device_allocation_type(i, DevAllocType::kNone);
}

template <typename T>
T LaunchContextBuilder::get_ret(int i) {
  return taichi_union_cast_with_different_sizes<T>(ctx_->result_buffer[i]);
}

#define PER_C_TYPE(type, ctype)                                            \
  template void LaunchContextBuilder::set_struct_arg_impl(                 \
      std::vector<int> arg_indices, ctype v);                              \
  template ctype LaunchContextBuilder::get_arg(const std::vector<int> &i); \
  template ctype LaunchContextBuilder::get_struct_arg(                     \
      std::vector<int> arg_indices);                                       \
  template void LaunchContextBuilder::set_arg(const std::vector<int> &i,   \
                                              ctype v);                    \
  template ctype LaunchContextBuilder::get_ret(int i);
#include "taichi/inc/data_type_with_c_type.inc.h"
PER_C_TYPE(gen, void *)  // Register void* as a valid type
#undef PER_C_TYPE

void LaunchContextBuilder::set_array_runtime_size(const std::vector<int> &i,
                                                  uint64 size) {
  array_runtime_sizes[i] = size;
}

void LaunchContextBuilder::set_array_device_allocation_type(
    const std::vector<int> &i,
    DevAllocType usage) {
  device_allocation_type[i] = usage;
}

void LaunchContextBuilder::set_arg_external_array_with_shape(
    const std::vector<int> &arg_id,
    uintptr_t ptr,
    uint64 size,
    const std::vector<int64> &shape,
    uintptr_t grad_ptr) {
  TI_ASSERT_INFO(
      kernel_->nested_parameters[arg_id].is_array,
      "Assigning external (numpy) array to scalar argument is not allowed.");

  TI_ASSERT_INFO(shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  array_ptrs[concatenate_vector<int>(
      arg_id, {TypeFactory::DATA_PTR_POS_IN_NDARRAY})] = (void *)ptr;
  array_ptrs[concatenate_vector<int>(
      arg_id, {TypeFactory::GRAD_PTR_POS_IN_NDARRAY})] = (void *)grad_ptr;
  set_array_runtime_size(arg_id, size);
  set_array_device_allocation_type(arg_id, DevAllocType::kNone);
  for (uint64 i = 0; i < shape.size(); ++i) {
    set_struct_arg(concatenate_vector<int>(arg_id, {0, (int32)i}),
                   (int32)shape[i]);
  }
}

void LaunchContextBuilder::set_arg_ndarray(const std::vector<int> &arg_id,
                                           const Ndarray &arr) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, ptr, arr.shape);
}

void LaunchContextBuilder::set_arg_argpack(const std::vector<int> &arg_id,
                                           const ArgPack &argpack) {
  argpack_ptrs[arg_id] = &argpack;
  if (arg_id.size() == 1) {
    // Only set ptr to arg buffer if this argpack is not nested
    set_argpack_ptr(arg_id, argpack.get_device_allocation_ptr_as_int());
  }
  // TODO: Consider renaming this method to `set_device_allocation_type`
  set_array_device_allocation_type(arg_id, DevAllocType::kArgPack);
}

void LaunchContextBuilder::set_arg_ndarray_with_grad(
    const std::vector<int> &arg_id,
    const Ndarray &arr,
    const Ndarray &arr_grad) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  intptr_t ptr_grad = arr_grad.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, ptr, arr.shape, ptr_grad);
}

void LaunchContextBuilder::set_arg_texture(const std::vector<int> &arg_id,
                                           const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  set_arg_texture_impl(arg_id, ptr);
}

void LaunchContextBuilder::set_arg_rw_texture(const std::vector<int> &arg_id,
                                              const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  set_arg_rw_texture_impl(arg_id, ptr, tex.get_size());
}

RuntimeContext &LaunchContextBuilder::get_context() {
  return *ctx_;
}

void LaunchContextBuilder::set_arg_texture_impl(const std::vector<int> &arg_id,
                                                intptr_t alloc_ptr) {
  array_ptrs[arg_id] = (void *)alloc_ptr;
  set_array_device_allocation_type(arg_id, DevAllocType::kTexture);
}

void LaunchContextBuilder::set_arg_rw_texture_impl(
    const std::vector<int> &arg_id,
    intptr_t alloc_ptr,
    const std::array<int, 3> &shape) {
  array_ptrs[arg_id] = (void *)alloc_ptr;
  set_array_device_allocation_type(arg_id, DevAllocType::kRWTexture);
  TI_ASSERT(shape.size() <= taichi_max_num_indices);
  for (int i = 0; i < shape.size(); ++i) {
    set_struct_arg(concatenate_vector<int>(arg_id, {0, i}), shape[i]);
  }
}

void LaunchContextBuilder::set_arg_ndarray_impl(const std::vector<int> &arg_id,
                                                intptr_t devalloc_ptr,
                                                const std::vector<int> &shape,
                                                intptr_t devalloc_ptr_grad) {
  // Set array ptr
  array_ptrs[concatenate_vector<int>(
      arg_id, {TypeFactory::DATA_PTR_POS_IN_NDARRAY})] = (void *)devalloc_ptr;
  if (devalloc_ptr != 0) {
    array_ptrs[concatenate_vector<int>(
        arg_id, {TypeFactory::GRAD_PTR_POS_IN_NDARRAY})] =
        (void *)devalloc_ptr_grad;
  }
  // Set device allocation type and runtime size
  set_array_device_allocation_type(arg_id, DevAllocType::kNdarray);
  TI_ASSERT(shape.size() <= taichi_max_num_indices);
  size_t total_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    set_struct_arg(concatenate_vector<int>(arg_id, {0, (int32)i}),
                   (int32)shape[i]);
    total_size *= shape[i];
  }
  set_array_runtime_size(arg_id, total_size);
}

void LaunchContextBuilder::set_arg_matrix(int arg_id, const Matrix &matrix) {
  int type_size = data_type_size(matrix.dtype());
  for (uint32_t i = 0; i < matrix.length(); i++) {
    switch (type_size) {
      case 1:
        set_struct_arg_impl({arg_id, (int32)i},
                            taichi_union_cast_with_different_sizes<int8>(
                                reinterpret_cast<uint8_t *>(matrix.data())[i]));
        break;
      case 2:
        set_struct_arg_impl(
            {arg_id, (int32)i},
            taichi_union_cast_with_different_sizes<int16>(
                reinterpret_cast<uint16_t *>(matrix.data())[i]));
        break;
      case 4:
        set_struct_arg_impl(
            {arg_id, (int32)i},
            taichi_union_cast_with_different_sizes<int32>(
                reinterpret_cast<uint32_t *>(matrix.data())[i]));
        break;
      case 8:
        set_struct_arg_impl(
            {arg_id, (int32)i},
            taichi_union_cast_with_different_sizes<int64>(
                reinterpret_cast<uint64_t *>(matrix.data())[i]));
        break;
      default:
        TI_ERROR("Unsupported type size {}", type_size);
    }
  }
}

TypedConstant LaunchContextBuilder::fetch_ret(const std::vector<int> &index) {
  const Type *dt = ret_type_->get_element_type(index);
  int offset = ret_type_->get_element_offset(index);
  return fetch_ret_impl(offset, dt);
}

float64 LaunchContextBuilder::get_struct_ret_float(
    const std::vector<int> &index) {
  return fetch_ret(index).val_float();
}

int64 LaunchContextBuilder::get_struct_ret_int(const std::vector<int> &index) {
  return fetch_ret(index).val_int();
}

uint64 LaunchContextBuilder::get_struct_ret_uint(
    const std::vector<int> &index) {
  return fetch_ret(index).val_uint();
}

TypedConstant LaunchContextBuilder::fetch_ret_impl(int offset, const Type *dt) {
  TI_ASSERT(dt->is<PrimitiveType>());
  auto primitive_type = dt->as<PrimitiveType>();
  char *ptr = result_buffer_.get() + offset;
  switch (primitive_type->type) {
#define PER_C_TYPE(type, ctype) \
  case PrimitiveTypeID::type:   \
    return TypedConstant(*(ctype *)ptr);
#include "taichi/inc/data_type_with_c_type.inc.h"
#undef PER_C_TYPE
    case PrimitiveTypeID::f16: {
      // first fetch the data as u16, and then convert it to f32
      uint16 half = *(uint16 *)ptr;
      return TypedConstant(fp16_ieee_to_fp32_value(half));
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

}  // namespace taichi::lang
