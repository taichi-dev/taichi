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
#include "taichi/util/statistics.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#endif

TLANG_NAMESPACE_BEGIN

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
  this->init(program, std::bind(func, this), primal_name, autodiff_mode);
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
  this->ir->as<Block>()->kernel = this;

  arch = program.config.arch;

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }

  if (!program.config.lazy_compilation)
    compile();
}

void Kernel::compile() {
  CurrentCallableGuard _(program, this);
  compiled_ = program->compile(*this);
}

void Kernel::compile_to_aot_kernel() {
  compiled_aot_kernel_ = program->make_aot_kernel(*this);
}

void Kernel::lower(bool to_executable) {
  TI_ASSERT(!lowered_);
  TI_ASSERT(supports_lowering(arch));

  CurrentCallableGuard _(program, this);
  auto config = program->config;
  bool verbose = config.print_ir;
  if ((is_accessor && !config.print_accessor_ir) ||
      (is_evaluator && !config.print_evaluator_ir))
    verbose = false;

  if (config.print_preprocessed_ir) {
    TI_INFO("[{}] {}:", get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir.get());
    irpass::print(ir.get());
    std::cout << std::flush;
  }

  if (to_executable) {
    irpass::compile_to_executable(
        ir.get(), config, this, /*autodiff_mode=*/autodiff_mode,
        /*ad_use_stack=*/true, verbose,
        /*lower_global_access=*/to_executable,
        /*make_thread_local=*/config.make_thread_local,
        /*make_block_local=*/
        is_extension_supported(config.arch, Extension::bls) &&
            config.make_block_local,
        /*start_from_ast=*/ir_is_ast_);
  } else {
    irpass::compile_to_offloads(ir.get(), config, this, verbose,
                                /*autodiff_mode=*/autodiff_mode,
                                /*ad_use_stack=*/true,
                                /*start_from_ast=*/ir_is_ast_);
  }

  lowered_ = true;
}

void Kernel::operator()(LaunchContextBuilder &ctx_builder) {
  if (!compiled_) {
    compile();
  }

  if (!this->from_offline_cache_) {
    for (auto &offloaded : ir->as<Block>()->statements) {
      account_for_offloaded(offloaded->as<OffloadedStmt>());
    }
  }

  compiled_(ctx_builder.get_context());

  program->sync = (program->sync && arch_is_cpu(arch));
  // Note that Kernel::arch may be different from program.config.arch
  if (program->config.debug && (arch_is_cpu(program->config.arch) ||
                                program->config.arch == Arch::cuda)) {
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
  TI_ASSERT_INFO(!kernel_->args[arg_id].is_array,
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_float64",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
       ActionArg("val", d)});

  auto dt = kernel_->args[arg_id].dt;
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
  TI_ASSERT_INFO(!kernel_->args[arg_id].is_array,
                 "Assigning scalar value to external (numpy) array argument is "
                 "not allowed.");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_int64",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
       ActionArg("val", d)});

  auto dt = kernel_->args[arg_id].dt;
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

void Kernel::LaunchContextBuilder::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

void Kernel::LaunchContextBuilder::set_arg_external_array_with_shape(
    int arg_id,
    uintptr_t ptr,
    uint64 size,
    const std::vector<int64> &shape) {
  TI_ASSERT_INFO(
      kernel_->args[arg_id].is_array,
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

void Kernel::LaunchContextBuilder::set_arg_texture(int arg_id,
                                                   const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg(arg_id, ptr);
  ctx_->set_array_device_allocation_type(
      arg_id, RuntimeContext::DevAllocType::kTexture);
}

void Kernel::LaunchContextBuilder::set_arg_rw_texture(int arg_id,
                                                      const Texture &tex) {
  intptr_t ptr = tex.get_device_allocation_ptr_as_int();
  ctx_->set_arg(arg_id, ptr);
  ctx_->set_array_device_allocation_type(
      arg_id, RuntimeContext::DevAllocType::kRWTexture);
}

void Kernel::LaunchContextBuilder::set_arg_raw(int arg_id, uint64 d) {
  TI_ASSERT_INFO(!kernel_->args[arg_id].is_array,
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

std::vector<int64> Kernel::get_ret_int_tensor(int i) {
  DataType dt = rets[i].dt->as<TensorType>()->get_element_type();
  int size = rets[i].dt->as<TensorType>()->get_num_elements();
  std::vector<int64> res;
  for (int j = 0; j < size; j++) {
    res.emplace_back(fetch_ret<int64>(dt, j));
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

void Kernel::set_arch(Arch arch) {
  TI_ASSERT(!compiled_);
  this->arch = arch;
}

void Kernel::account_for_offloaded(OffloadedStmt *stmt) {
  if (is_evaluator || is_accessor)
    return;
  auto task_type = stmt->task_type;
  stat.add("launched_tasks", 1.0);
  if (task_type == OffloadedStmt::TaskType::listgen) {
    stat.add("launched_tasks_list_op", 1.0);
    stat.add("launched_tasks_list_gen", 1.0);
  } else if (task_type == OffloadedStmt::TaskType::serial) {
    // TODO: Do we need to distinguish serial tasks that contain clear lists vs
    // those who don't?
    stat.add("launched_tasks_compute", 1.0);
    stat.add("launched_tasks_serial", 1.0);
  } else if (task_type == OffloadedStmt::TaskType::range_for) {
    stat.add("launched_tasks_compute", 1.0);
    stat.add("launched_tasks_range_for", 1.0);
  } else if (task_type == OffloadedStmt::TaskType::struct_for) {
    stat.add("launched_tasks_compute", 1.0);
    stat.add("launched_tasks_struct_for", 1.0);
  } else if (task_type == OffloadedStmt::TaskType::mesh_for) {
    stat.add("launched_tasks_compute", 1.0);
    stat.add("launched_tasks_mesh_for", 1.0);
  } else if (task_type == OffloadedStmt::TaskType::gc) {
    stat.add("launched_tasks_garbage_collect", 1.0);
  }
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
  context = std::make_unique<FrontendContext>(program.config.arch);
  ir = context->get_root();
  ir_is_ast_ = true;

  this->arch = program.config.arch;

  if (autodiff_mode == AutodiffMode::kNone) {
    name = primal_name;
  } else if (autodiff_mode == AutodiffMode::kForward) {
    name = primal_name + "_forward_grad";
  } else if (autodiff_mode == AutodiffMode::kReverse) {
    name = primal_name + "_reverse_grad";
  }

  {
    // Note: this is NOT a mutex. If we want to call Kernel::Kernel()
    // concurrently, we need to lock this block of code together with
    // taichi::lang::context with a mutex.
    CurrentCallableGuard _(this->program, this);
    func();
    ir->as<Block>()->kernel = this;
  }

  if (!program.config.lazy_compilation)
    compile();
}

// static
bool Kernel::supports_lowering(Arch arch) {
  return arch_is_cpu(arch) || (arch == Arch::cuda) || (arch == Arch::metal);
}

void Kernel::offload_to_executable(IRNode *stmt) {
  CurrentCallableGuard _(program, this);
  auto config = program->config;
  bool verbose = config.print_ir;
  if ((is_accessor && !config.print_accessor_ir) ||
      (is_evaluator && !config.print_evaluator_ir))
    verbose = false;
  irpass::offload_to_executable(
      stmt, config, this, verbose,
      /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse,
      /*lower_global_access=*/true,
      /*make_block_local=*/config.make_thread_local,
      /*make_block_local=*/
      is_extension_supported(config.arch, Extension::bls) &&
          config.make_block_local);
}
TLANG_NAMESPACE_END
