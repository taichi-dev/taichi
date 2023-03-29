#include <taichi/program/launch_context_builder.h>
#define TI_RUNTIME_HOST
#include <taichi/program/context.h>
#undef TI_RUNTIME_HOST
#include "taichi/util/action_recorder.h"
#include "fp16.h"

namespace taichi::lang {

LaunchContextBuilder::LaunchContextBuilder(CallableBase *kernel)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<RuntimeContext>()),
      ctx_(owned_ctx_.get()),
      arg_buffer_(std::make_unique<char[]>(
          arch_uses_llvm(kernel->arch)
              ? kernel->args_size
              : sizeof(uint64) * taichi_max_num_args_total)),
      result_buffer_(std::make_unique<char[]>(
          arch_uses_llvm(kernel->arch)
              ? kernel->ret_size
              : sizeof(uint64) * taichi_result_buffer_entries)),
      ret_type_(kernel->ret_type),
      arg_buffer_size(kernel->args_size),
      args_type(kernel->args_type),
      result_buffer_size(kernel->ret_size) {
  ctx_->result_buffer = (uint64 *)result_buffer_.get();
  ctx_->arg_buffer = arg_buffer_.get();
}

void LaunchContextBuilder::set_arg_float(int arg_id, float64 d) {
  TI_ASSERT_INFO(!kernel_->parameter_list[arg_id].is_array,
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_float64",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
       ActionArg("val", d)});

  auto dt = kernel_->parameter_list[arg_id].get_dtype();
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    set_arg(arg_id, (float32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    set_arg(arg_id, (float64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    set_arg(arg_id, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    set_arg(arg_id, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    set_arg(arg_id, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    set_arg(arg_id, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    set_arg(arg_id, (uint64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    if (!arch_uses_llvm(kernel_->arch)) {
      // TODO: remove this once we refactored the SPIR-V based backends
      set_arg(arg_id, (float32)d);
      return;
    }
    uint16 half = fp16_ieee_from_fp32_value((float32)d);
    set_arg(arg_id, half);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_arg_int(int arg_id, int64 d) {
  TI_ASSERT_INFO(!kernel_->parameter_list[arg_id].is_array,
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_integer",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
       ActionArg("val", d)});

  auto dt = kernel_->parameter_list[arg_id].get_dtype();
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    set_arg(arg_id, (int16)d);
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

void LaunchContextBuilder::set_arg_uint(int arg_id, uint64 d) {
  set_arg_int(arg_id, d);
}

template <>
void LaunchContextBuilder::set_arg<TypedConstant>(int i, TypedConstant d) {
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

void LaunchContextBuilder::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

template <typename T>
void LaunchContextBuilder::set_struct_arg(std::vector<int> index, T v) {
  if (!arch_uses_llvm(kernel->arch)) {
    return;
  }
  int offset = args_type->get_element_offset(index);
  *(T *)(ctx_->arg_buffer + offset) = v;
}

template <typename T>
T LaunchContextBuilder::get_arg(int i) {
  if (arch_uses_llvm(kernel->arch)) {
    return get_struct_arg<T>({i});
  }
  return taichi_union_cast_with_different_sizes<T>(ctx_->args[i]);
}

template <typename T>
T LaunchContextBuilder::get_struct_arg(std::vector<int> index) {
  int offset = args_type->get_element_offset(index);
  return *(T *)(ctx_->arg_buffer + offset);
}

template <typename T>
T LaunchContextBuilder::get_grad_arg(int i) {
  return taichi_union_cast_with_different_sizes<T>(ctx_->grad_args[i]);
}

template <typename T>
void LaunchContextBuilder::set_arg(int i, T v) {
  set_struct_arg({i}, v);
  ctx_->args[i] = taichi_union_cast_with_different_sizes<uint64>(v);
  set_array_device_allocation_type(i, DevAllocType::kNone);
}

template <typename T>
void LaunchContextBuilder::set_grad_arg(int i, T v) {
  ctx_->grad_args[i] = taichi_union_cast_with_different_sizes<uint64>(v);
}

template <typename T>
T LaunchContextBuilder::get_ret(int i) {
  return taichi_union_cast_with_different_sizes<T>(ctx_->result_buffer[i]);
}

#define PER_C_TYPE(type, ctype)                                                \
  template void LaunchContextBuilder::set_struct_arg(std::vector<int> index,   \
                                                     ctype v);                 \
  template ctype LaunchContextBuilder::get_arg(int i);                         \
  template ctype LaunchContextBuilder::get_struct_arg(std::vector<int> index); \
  template ctype LaunchContextBuilder::get_grad_arg(int i);                    \
  template void LaunchContextBuilder::set_arg(int i, ctype v);                 \
  template void LaunchContextBuilder::set_grad_arg(int i, ctype v);            \
  template ctype LaunchContextBuilder::get_ret(int i);
#include "taichi/inc/data_type_with_c_type.inc.h"
PER_C_TYPE(gen, void *)  // Register void* as a valid type
#undef PER_C_TYPE

void LaunchContextBuilder::set_array_runtime_size(int i, uint64 size) {
  array_runtime_sizes[i] = size;
}

void LaunchContextBuilder::set_array_device_allocation_type(
    int i,
    DevAllocType usage) {
  device_allocation_type[i] = usage;
}

void LaunchContextBuilder::set_arg_external_array_with_shape(
    int arg_id,
    uintptr_t ptr,
    uint64 size,
    const std::vector<int64> &shape) {
  TI_ASSERT_INFO(
      kernel_->parameter_list[arg_id].is_array,
      "Assigning external (numpy) array to scalar argument is not allowed.");

  TI_ASSERT_INFO(shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  array_ptrs[{arg_id}] = (void *)ptr;
  set_array_runtime_size(arg_id, size);
  set_array_device_allocation_type(arg_id, DevAllocType::kNone);
  for (uint64 i = 0; i < shape.size(); ++i) {
    ctx_->extra_args[arg_id][i] = shape[i];
  }
}

void LaunchContextBuilder::set_arg_ndarray(int arg_id, const Ndarray &arr) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, ptr, arr.shape);
}

void LaunchContextBuilder::set_arg_ndarray_with_grad(int arg_id,
                                                     const Ndarray &arr,
                                                     const Ndarray &arr_grad) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  intptr_t ptr_grad = arr_grad.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  set_arg_ndarray_impl(arg_id, ptr, arr.shape, true, ptr_grad);
}

void LaunchContextBuilder::set_arg_texture(int arg_id, const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  set_arg_texture_impl(arg_id, ptr);
}

void LaunchContextBuilder::set_arg_rw_texture(int arg_id, const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  set_arg_rw_texture_impl(arg_id, ptr, tex.get_size());
}

RuntimeContext &LaunchContextBuilder::get_context() {
  return *ctx_;
}

void LaunchContextBuilder::set_arg_texture_impl(int arg_id,
                                                intptr_t alloc_ptr) {
  array_ptrs[{arg_id}] = (void *)alloc_ptr;
  set_array_device_allocation_type(arg_id, DevAllocType::kTexture);
}

void LaunchContextBuilder::set_arg_rw_texture_impl(
    int arg_id,
    intptr_t alloc_ptr,
    const std::array<int, 3> &shape) {
  array_ptrs[{arg_id}] = (void *)alloc_ptr;
  set_array_device_allocation_type(arg_id, DevAllocType::kRWTexture);
  TI_ASSERT(shape.size() <= taichi_max_num_indices);
  for (int i = 0; i < shape.size(); i++) {
    ctx_->extra_args[arg_id][i] = shape[i];
  }
}

void LaunchContextBuilder::set_arg_ndarray_impl(int arg_id,
                                                intptr_t devalloc_ptr,
                                                const std::vector<int> &shape,
                                                bool grad,
                                                intptr_t devalloc_ptr_grad) {
  // Set has_grad value
  has_grad[arg_id] = grad;

  // Set array ptr
  array_ptrs[{arg_id}] = (void *)devalloc_ptr;

  // Set grad_args[arg_id] value
  if (grad) {
    ctx_->grad_args[arg_id] =
        taichi_union_cast_with_different_sizes<uint64>(devalloc_ptr_grad);
  }

  // Set device allocation type and runtime size
  set_array_device_allocation_type(arg_id, DevAllocType::kNdarray);
  TI_ASSERT(shape.size() <= taichi_max_num_indices);
  size_t total_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    ctx_->extra_args[arg_id][i] = shape[i];
    total_size *= shape[i];
  }
  set_array_runtime_size(arg_id, total_size);
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
