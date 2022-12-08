#include "taichi/program/kernel.h"

#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/logging.h"
#include "taichi/common/task.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/extension.h"
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

KernelLaunchContext Kernel::make_launch_context() {
  return KernelLaunchContext(this);
}

KernelLaunchContext::KernelLaunchContext(Kernel *kernel, RuntimeContext *ctx)
    : kernel_(kernel), owned_ctx_(nullptr), ctx_(ctx) {
}

KernelLaunchContext::KernelLaunchContext(Kernel *kernel)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<RuntimeContext>()),
      ctx_(owned_ctx_.get()) {
}

void KernelLaunchContext::set_arg_float(int arg_id, float64 d) {
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

void KernelLaunchContext::set_arg_int(int arg_id, int64 d) {
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

void KernelLaunchContext::set_arg_uint(int arg_id, uint64 d) {
  set_arg_int(arg_id, d);
}

void KernelLaunchContext::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

void KernelLaunchContext::set_arg_external_array_with_shape(
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

void KernelLaunchContext::set_arg_ndarray(int arg_id, const Ndarray &arr) {
  intptr_t ptr = arr.get_device_allocation_ptr_as_int();
  TI_ASSERT_INFO(arr.shape.size() <= taichi_max_num_indices,
                 "External array cannot have > {max_num_indices} indices");
  ctx_->set_arg_ndarray(arg_id, ptr, arr.shape);
}

void KernelLaunchContext::set_arg_texture(int arg_id, const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg_texture(arg_id, ptr);
}

void KernelLaunchContext::set_arg_rw_texture(int arg_id, const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg_rw_texture(arg_id, ptr, tex.get_size());
}

void KernelLaunchContext::set_arg_raw(int arg_id, uint64 d) {
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

// Refactor2023:FIXME: Bad smell. Use template function.
float64 KernelLaunchContext::get_ret_float(Device *device,
                                           unsigned retNo) const {
  auto *dt = kernel_->rets[retNo].dt->get_compute_type();
  return fetch_ret<float64>(dt, retNo, device, ctx_);
}

uint64 KernelLaunchContext::get_ret_raw(Device *device, unsigned retNo) const {
  return device->fetch_result_uint64(retNo, ctx_->result_buffer);
}

int64 KernelLaunchContext::get_ret_int(Device *device, unsigned retNo) const {
  auto *dt = kernel_->rets[retNo].dt->get_compute_type();
  auto p = fetch_ret<int64>(dt, retNo, device, ctx_);
  return p;
}

uint64 KernelLaunchContext::get_ret_uint(Device *device, unsigned retNo) const {
  auto *dt = kernel_->rets[retNo].dt->get_compute_type();
  return fetch_ret<uint64>(dt, retNo, device, ctx_);
}

std::vector<int64> KernelLaunchContext::get_ret_int_tensor(
    Device *device,
    unsigned retNo) const {
  auto *tensor_dt = kernel_->rets[retNo].dt->as<TensorType>();
  TI_ASSERT(tensor_dt != nullptr);
  DataType element_dt = tensor_dt->get_element_type();
  int element_count = tensor_dt->get_num_elements();
  TI_ASSERT(element_count >= 0);
  std::vector<int64> res;
  for (unsigned j = 0; j < (unsigned)element_count; ++j) {
    res.push_back(fetch_ret<int64>(element_dt, j, device, ctx_));
  }
  return res;
}

std::vector<uint64> KernelLaunchContext::get_ret_uint_tensor(
    Device *device,
    unsigned retNo) const {
  auto *tensor_dt = kernel_->rets[retNo].dt->as<TensorType>();
  TI_ASSERT(tensor_dt != nullptr);
  DataType element_dt = tensor_dt->get_element_type();
  int element_count = tensor_dt->get_num_elements();
  TI_ASSERT(element_count >= 0);
  std::vector<uint64> res;
  for (unsigned j = 0; j < (unsigned)element_count; ++j) {
    res.push_back(fetch_ret<uint64>(element_dt, j, device, ctx_));
  }
  return res;
}

std::vector<float64> KernelLaunchContext::get_ret_float_tensor(
    Device *device,
    unsigned retNo) const {
  auto *tensor_dt = kernel_->rets[retNo].dt->as<TensorType>();
  TI_ASSERT(tensor_dt != nullptr);
  DataType element_dt = tensor_dt->get_element_type();
  int element_count = tensor_dt->get_num_elements();
  TI_ASSERT(element_count >= 0);
  std::vector<float64> res;
  for (unsigned j = 0; j < (unsigned)element_count; ++j) {
    res.push_back(fetch_ret<float64>(element_dt, j, device, ctx_));
  }
  return res;
}

RuntimeContext &KernelLaunchContext::get_context() {
  // Refactor2023:FIXME: Move to KernelLauncher
  kernel_->program->prepare_runtime_context(ctx_);
  return *ctx_;
}

template <typename T>
T KernelLaunchContext::fetch_ret(DataType dt,
                                 unsigned retNo,
                                 Device *device,
                                 RuntimeContext *rt_ctx) {
  TI_ASSERT(device);

  auto *primative_dt = dt->cast<PrimitiveType>();
  if (!primative_dt) {
    TI_NOT_IMPLEMENTED;
  }

#define FETCH_AND_CAST(dt_enum, dt_type)                                \
  case dt_enum: {                                                       \
    auto i = device->fetch_result_uint64(retNo, rt_ctx->result_buffer); \
    return (T)taichi_union_cast_with_different_sizes<dt_type>(i);       \
  }

  switch (primative_dt->type) {
    FETCH_AND_CAST(PrimitiveTypeID::f32, float32);
    FETCH_AND_CAST(PrimitiveTypeID::f64, float64);
    FETCH_AND_CAST(PrimitiveTypeID::i32, int32);
    FETCH_AND_CAST(PrimitiveTypeID::i64, int64);
    FETCH_AND_CAST(PrimitiveTypeID::i8, int8);
    FETCH_AND_CAST(PrimitiveTypeID::i16, int16);
    FETCH_AND_CAST(PrimitiveTypeID::u8, uint8);
    FETCH_AND_CAST(PrimitiveTypeID::u16, uint16);
    FETCH_AND_CAST(PrimitiveTypeID::u32, uint32);
    FETCH_AND_CAST(PrimitiveTypeID::u64, uint64);
    FETCH_AND_CAST(PrimitiveTypeID::f16, float32);  // use f32
    default:
      TI_NOT_IMPLEMENTED;
  }
#undef FETCH_AND_CAST
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
  context =
      std::make_unique<FrontendContext>(program.this_thread_config().arch);
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

// static
bool Kernel::supports_lowering(Arch arch) {
  return arch_is_cpu(arch) || (arch == Arch::cuda) || (arch == Arch::dx12) ||
         (arch == Arch::metal);
}

void Kernel::offload_to_executable(IRNode *stmt) {
  auto config = program->this_thread_config();
  bool verbose = config.print_ir;
  if ((is_accessor && !config.print_accessor_ir) ||
      (is_evaluator && !config.print_evaluator_ir))
    verbose = false;
  irpass::offload_to_executable(
      stmt, config, this, verbose,
      /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse,
      /*lower_global_access=*/true,
      /*make_thread_local=*/config.make_thread_local,
      /*make_block_local=*/
      is_extension_supported(config.arch, Extension::bls) &&
          config.make_block_local);
}

// Refactor2023:FIXME: Remove (:Temp)
void launch_kernel(Program *prog,
                   const CompileConfig &compile_config,
                   Kernel &kernel,
                   RuntimeContext &ctx) {
  auto fn = kernel.get_compiled_func();
  if (!fn) {
    kernel.set_compiled_func(fn = prog->compile(compile_config, kernel));
  }

  TI_ASSERT(!!fn);
  fn(ctx);

  const auto arch = compile_config.arch;
  prog->sync = (prog->sync && arch_is_cpu(arch));
  if (compile_config.debug && (arch_is_cpu(arch) || arch == Arch::cuda)) {
    prog->check_runtime_error();
  }
}
}  // namespace taichi::lang
