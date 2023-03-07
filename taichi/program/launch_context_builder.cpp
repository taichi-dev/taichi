#include <taichi/program/launch_context_builder.h>
#define TI_RUNTIME_HOST
#include <taichi/program/context.h>
#undef TI_RUNTIME_HOST
#include "taichi/util/action_recorder.h"

namespace taichi::lang {
LaunchContextBuilder::LaunchContextBuilder(CallableBase *kernel,
                                           RuntimeContext *ctx)
    : kernel_(kernel),
      owned_ctx_(nullptr),
      ctx_(ctx),
      arg_buffer_(std::make_unique<char[]>(
          arch_uses_llvm(kernel->arch)
              ? kernel->args_size
              : sizeof(uint64) * taichi_max_num_args_total)),
      result_buffer_(std::make_unique<char[]>(
          arch_uses_llvm(kernel->arch)
              ? kernel->ret_size
              : sizeof(uint64) * taichi_result_buffer_entries)) {
  if (arch_uses_llvm(kernel->arch)) {
    ctx_->result_buffer = (uint64 *)result_buffer_.get();
    ctx_->result_buffer_size = kernel->ret_size;
    ctx_->arg_buffer_size = kernel->args_size;
    ctx_->arg_buffer = arg_buffer_.get();
    ctx_->args_type = kernel->args_type;
  }
}

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
      ret_type_(kernel->ret_type) {
  ctx_->result_buffer = (uint64 *)result_buffer_.get();
  ctx_->result_buffer_size = kernel->ret_size;
  ctx_->arg_buffer_size = kernel->args_size;
  ctx_->arg_buffer = arg_buffer_.get();
  ctx_->args_type = kernel->args_type;
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
    ctx_->set_arg(arg_id, (float32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    ctx_->set_arg(arg_id, (float64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    ctx_->set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    ctx_->set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    ctx_->set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    ctx_->set_arg(arg_id, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    ctx_->set_arg(arg_id, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    ctx_->set_arg(arg_id, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    ctx_->set_arg(arg_id, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    ctx_->set_arg(arg_id, (uint64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // use f32 to interact with python
    ctx_->set_arg(arg_id, (float32)d);
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
    ctx_->set_arg(arg_id, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    ctx_->set_arg(arg_id, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    ctx_->set_arg(arg_id, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    ctx_->set_arg(arg_id, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    ctx_->set_arg(arg_id, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    ctx_->set_arg(arg_id, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    ctx_->set_arg(arg_id, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    ctx_->set_arg(arg_id, (uint64)d);
  } else {
    TI_INFO(dt->to_string());
    TI_NOT_IMPLEMENTED
  }
}

void LaunchContextBuilder::set_arg_uint(int arg_id, uint64 d) {
  set_arg_int(arg_id, d);
}

void LaunchContextBuilder::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

void LaunchContextBuilder::set_arg_external_array_with_shape(
    int arg_id,
    uintptr_t ptr,
    uint64 size,
    const std::vector<int64> &shape) {
  TI_ASSERT_INFO(
      kernel_->parameter_list[arg_id].is_array,
      "Assigning external (numpy) array to scalar argument is not allowed.");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_ext_ptr",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
       ActionArg("address", fmt::format("0x{:x}", ptr)),
       ActionArg("array_size_in_bytes", (int64)size)});

  TI_ASSERT_INFO(shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  ctx_->set_arg_external_array(arg_id, ptr, size, shape);
}

void LaunchContextBuilder::set_arg_ndarray(int arg_id, const Ndarray &arr) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  ctx_->set_arg_ndarray(arg_id, ptr, arr.shape);
}

void LaunchContextBuilder::set_arg_ndarray_with_grad(int arg_id,
                                                     const Ndarray &arr,
                                                     const Ndarray &arr_grad) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  intptr_t ptr_grad = arr_grad.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  ctx_->set_arg_ndarray(arg_id, ptr, arr.shape, true, ptr_grad);
}

void LaunchContextBuilder::set_arg_texture(int arg_id, const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg_texture(arg_id, ptr);
}

void LaunchContextBuilder::set_arg_rw_texture(int arg_id, const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg_rw_texture(arg_id, ptr, tex.get_size());
}

void LaunchContextBuilder::set_arg_raw(int arg_id, uint64 d) {
  TI_ASSERT_INFO(!kernel_->parameter_list[arg_id].is_array,
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  ctx_->set_arg<uint64>(arg_id, d);
}

RuntimeContext &LaunchContextBuilder::get_context() {
  return *ctx_;
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
#define RETURN_PRIMITIVE(type, ctype) \
  case PrimitiveTypeID::type:         \
    return TypedConstant(*(ctype *)ptr);

    RETURN_PRIMITIVE(f32, float32);
    RETURN_PRIMITIVE(f64, float64);
    RETURN_PRIMITIVE(i8, int8);
    RETURN_PRIMITIVE(i16, int16);
    RETURN_PRIMITIVE(i32, int32);
    RETURN_PRIMITIVE(i64, int64);
    RETURN_PRIMITIVE(u8, uint8);
    RETURN_PRIMITIVE(u16, uint16);
    RETURN_PRIMITIVE(u32, uint32);
    RETURN_PRIMITIVE(u64, uint64);
#undef RETURN_PRIMITIVE
    case PrimitiveTypeID::f16: {
      // first fetch the data as u16, and then convert it to f32
      uint16 half = *(uint16 *)ptr;
      return TypedConstant(bit::half_to_float(half));
    }
    default:
      TI_NOT_IMPLEMENTED
  }
}

}  // namespace taichi::lang
