#include "taichi/program/kernel.h"

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/logging.h"
#include "taichi/common/task.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/action_recorder.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

namespace taichi::lang {

class Function;

Kernel::Kernel(Program &program,
               const std::function<void()> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  this->init(program, func, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               const std::function<void(Kernel *)> &func,
               const std::string &primal_name,
               AutodiffMode autodiff_mode) {
  // due to #6362, we cannot write [func, this] { return func(this); }
  this->init(
      program, [&] { return func(this); }, primal_name, autodiff_mode);
}

Kernel::Kernel(Program &program,
               std::unique_ptr<IRNode> &&ir,
               const std::string &primal_name,
               AutodiffMode autodiff_mode)
    : autodiff_mode(autodiff_mode), lowered_(false) {
  this->ir = std::move(ir);
  this->program = &program;
  is_accessor = false;
  is_evaluator = false;
  compiled_ = nullptr;
  ir_is_ast_ = false;  // CHI IR

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }
}

void Kernel::compile(const CompileConfig &compile_config) {
  compiled_ = program->compile(compile_config, *this);
}

void Kernel::operator()(const CompileConfig &compile_config,
                        LaunchContextBuilder &ctx_builder) {
  if (!compiled_) {
    compile(compile_config);
  }

  compiled_(ctx_builder.get_context());

  const auto arch = compile_config.arch;
  if (compile_config.debug && (arch_is_cpu(arch) || arch == Arch::cuda)) {
    program->check_runtime_error();
  }
}

Kernel::LaunchContextBuilder Kernel::make_launch_context() {
  return LaunchContextBuilder(this);
}

Kernel::LaunchContextBuilder::LaunchContextBuilder(Kernel *kernel,
                                                   RuntimeContext *ctx)
    : kernel_(kernel), owned_ctx_(nullptr), ctx_(ctx) {
}

Kernel::LaunchContextBuilder::LaunchContextBuilder(Kernel *kernel)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<RuntimeContext>()),
      ctx_(owned_ctx_.get()) {
}

void Kernel::LaunchContextBuilder::set_arg_float(int arg_id, float64 d) {
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

void Kernel::LaunchContextBuilder::set_arg_int(int arg_id, int64 d) {
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

void Kernel::LaunchContextBuilder::set_arg_uint(int arg_id, uint64 d) {
  set_arg_int(arg_id, d);
}

void Kernel::LaunchContextBuilder::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

void Kernel::LaunchContextBuilder::set_arg_external_array_with_shape(
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

void Kernel::LaunchContextBuilder::set_arg_ndarray(int arg_id,
                                                   const Ndarray &arr) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  ctx_->set_arg_ndarray(arg_id, ptr, arr.shape);
}

void Kernel::LaunchContextBuilder::set_arg_ndarray_with_grad(
    int arg_id,
    const Ndarray &arr,
    const Ndarray &arr_grad) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  intptr_t ptr_grad = arr_grad.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  ctx_->set_arg_ndarray(arg_id, ptr, arr.shape, true, ptr_grad);
}

void Kernel::LaunchContextBuilder::set_arg_texture(int arg_id,
                                                   const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg_texture(arg_id, ptr);
}

void Kernel::LaunchContextBuilder::set_arg_rw_texture(int arg_id,
                                                      const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg_rw_texture(arg_id, ptr, tex.get_size());
}

void Kernel::LaunchContextBuilder::set_arg_raw(int arg_id, uint64 d) {
  TI_ASSERT_INFO(!kernel_->parameter_list[arg_id].is_array,
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  if (!kernel_->is_evaluator) {
    ActionRecorder::get_instance().record(
        "set_arg_raw",
        {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
         ActionArg("val", (int64)d)});
  }
  ctx_->set_arg<uint64>(arg_id, d);
}

RuntimeContext &Kernel::LaunchContextBuilder::get_context() {
  kernel_->program->prepare_runtime_context(ctx_);
  return *ctx_;
}

template <typename T>
T Kernel::fetch_ret(DataType dt, int i) {
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (T)program->fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (T)program->fetch_result<float64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (T)program->fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (T)program->fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (T)program->fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (T)program->fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (T)program->fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (T)program->fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (T)program->fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (T)program->fetch_result<uint64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // use f32 to interact with python
    return (T)program->fetch_result<float32>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 Kernel::get_ret_float(int i) {
  auto dt = rets[i].dt->get_compute_type();
  return fetch_ret<float64>(dt, i);
}

int64 Kernel::get_ret_int(int i) {
  auto dt = rets[i].dt->get_compute_type();
  return fetch_ret<int64>(dt, i);
}

uint64 Kernel::get_ret_uint(int i) {
  auto dt = rets[i].dt->get_compute_type();
  return fetch_ret<uint64>(dt, i);
}

std::vector<int64> Kernel::get_ret_int_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<int64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<int64>(dt, j));
  }
  return res;
}

std::vector<uint64> Kernel::get_ret_uint_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<uint64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<uint64>(dt, j));
  }
  return res;
}

std::vector<float64> Kernel::get_ret_float_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<float64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<float64>(dt, j));
  }
  return res;
}

std::string Kernel::get_name() const {
  return name;
}

void Kernel::init(Program &program,
                  const std::function<void()> &func,
                  const std::string &primal_name,
                  AutodiffMode autodiff_mode) {
  this->autodiff_mode = autodiff_mode;
  this->lowered_ = false;
  this->program = &program;

  is_accessor = false;
  is_evaluator = false;
  compiled_ = nullptr;
  context = std::make_unique<FrontendContext>(program.compile_config().arch);
  ir = context->get_root();
  ir_is_ast_ = true;

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    name = primal_name + "_validate_grad";
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }

  func();
}
template <typename T>
T Kernel::fetch_ret(std::vector<int> index) {
  return nullptr;
}

}  // namespace taichi::lang
