#include "taichi/program/kernel.h"

#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/codegen/codegen.h"
#include "taichi/common/task.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/async_engine.h"
#include "taichi/program/extension.h"
#include "taichi/program/program.h"
#include "taichi/util/action_recorder.h"
#include "taichi/util/statistics.h"

TLANG_NAMESPACE_BEGIN

class Function;

Kernel::Kernel(Program &program,
               const std::function<void()> &func,
               const std::string &primal_name,
               bool grad)
    : grad(grad), lowered_(false) {
  this->program = &program;
  // Do not corrupt the context calling this kernel here -- maybe unnecessary
  auto backup_context = std::move(taichi::lang::context);

  program.maybe_initialize_cuda_llvm_context();
  is_accessor = false;
  is_evaluator = false;
  compiled_ = nullptr;
  taichi::lang::context = std::make_unique<FrontendContext>();
  ir = taichi::lang::context->get_root();
  ir_is_ast_ = true;

  {
    // Note: this is NOT a mutex. If we want to call Kernel::Kernel()
    // concurrently, we need to lock this block of code together with
    // taichi::lang::context with a mutex.
    CurrentCallableGuard _(this->program, this);
    program.start_kernel_definition(this);
    func();
    program.end_kernel_definition();
    ir->as<Block>()->kernel = this;
  }

  arch = program.config.arch;

  if (!grad) {
    name = primal_name;
  } else {
    name = primal_name + "_grad";
  }

  if (!program.config.lazy_compilation)
    compile();

  taichi::lang::context = std::move(backup_context);
}

Kernel::Kernel(Program &program,
               std::unique_ptr<IRNode> &&ir,
               const std::string &primal_name,
               bool grad)
    : grad(grad), lowered_(false) {
  this->ir = std::move(ir);
  this->program = &program;
  is_accessor = false;
  is_evaluator = false;
  compiled_ = nullptr;
  ir_is_ast_ = false;  // CHI IR
  this->ir->as<Block>()->kernel = this;

  arch = program.config.arch;

  if (!grad) {
    name = primal_name;
  } else {
    name = primal_name + "_grad";
  }

  if (!program.config.lazy_compilation)
    compile();
}

void Kernel::compile() {
  CurrentCallableGuard _(program, this);
  compiled_ = program->compile(*this);
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

  if (to_executable) {
    irpass::compile_to_executable(
        ir.get(), config, this, /*vectorize*/ arch_is_cpu(arch), grad,
        /*ad_use_stack=*/true, verbose, /*lower_global_access=*/to_executable,
        /*make_thread_local=*/config.make_thread_local,
        /*make_block_local=*/
        is_extension_supported(config.arch, Extension::bls) &&
            config.make_block_local,
        /*start_from_ast=*/ir_is_ast_);
  } else {
    irpass::compile_to_offloads(ir.get(), config, this, verbose,
                                /*vectorize=*/arch_is_cpu(arch), grad,
                                /*ad_use_stack=*/true,
                                /*start_from_ast=*/ir_is_ast_);
  }

  lowered_ = true;
}

void Kernel::operator()(LaunchContextBuilder &ctx_builder) {
  if (!program->config.async_mode || this->is_evaluator) {
    if (!compiled_) {
      compile();
    }

    for (auto &offloaded : ir->as<Block>()->statements) {
      account_for_offloaded(offloaded->as<OffloadedStmt>());
    }

    compiled_(ctx_builder.get_context());

    program->sync = (program->sync && arch_is_cpu(arch));
    // Note that Kernel::arch may be different from program.config.arch
    if (program->config.debug && (arch_is_cpu(program->config.arch) ||
                                  program->config.arch == Arch::cuda)) {
      program->check_runtime_error();
    }
  } else {
    program->sync = false;
    program->async_engine->launch(this, ctx_builder.get_context());
    // Note that Kernel::arch may be different from program.config.arch
    if (program->config.debug && arch_is_cpu(arch) &&
        arch_is_cpu(program->config.arch)) {
      program->check_runtime_error();
    }
  }
}

Kernel::LaunchContextBuilder Kernel::make_launch_context() {
  return LaunchContextBuilder(this);
}

Kernel::LaunchContextBuilder::LaunchContextBuilder(Kernel *kernel, Context *ctx)
    : kernel_(kernel), owned_ctx_(nullptr), ctx_(ctx) {
}

Kernel::LaunchContextBuilder::LaunchContextBuilder(Kernel *kernel)
    : kernel_(kernel),
      owned_ctx_(std::make_unique<Context>()),
      ctx_(owned_ctx_.get()) {
}

void Kernel::LaunchContextBuilder::set_arg_float(int arg_id, float64 d) {
  TI_ASSERT_INFO(!kernel_->args[arg_id].is_external_array,
                 "Assigning scalar value to external(numpy) array argument is "
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
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::LaunchContextBuilder::set_arg_int(int arg_id, int64 d) {
  TI_ASSERT_INFO(!kernel_->args[arg_id].is_external_array,
                 "Assigning scalar value to external(numpy) array argument is "
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
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    ctx_->set_arg(arg_id, (float32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    ctx_->set_arg(arg_id, (float64)d);
  } else {
    TI_INFO(dt->to_string());
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::LaunchContextBuilder::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

void Kernel::LaunchContextBuilder::set_arg_external_array(int arg_id,
                                                          uint64 ptr,
                                                          uint64 size) {
  TI_ASSERT_INFO(
      kernel_->args[arg_id].is_external_array,
      "Assigning external(numpy) array to scalar argument is not allowed.");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_ext_ptr",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
       ActionArg("address", fmt::format("0x{:x}", ptr)),
       ActionArg("array_size_in_bytes", (int64)size)});

  kernel_->args[arg_id].size = size;
  ctx_->set_arg(arg_id, ptr);
}

void Kernel::LaunchContextBuilder::set_arg_raw(int arg_id, uint64 d) {
  TI_ASSERT_INFO(!kernel_->args[arg_id].is_external_array,
                 "Assigning scalar value to external(numpy) array argument is "
                 "not allowed.");

  if (!kernel_->is_evaluator) {
    ActionRecorder::get_instance().record(
        "set_arg_raw",
        {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", arg_id),
         ActionArg("val", (int64)d)});
  }
  ctx_->set_arg<uint64>(arg_id, d);
}

Context &Kernel::LaunchContextBuilder::get_context() {
  ctx_->runtime = static_cast<LLVMRuntime *>(kernel_->program->llvm_runtime);
  return *ctx_;
}

float64 Kernel::get_ret_float(int i) {
  auto dt = rets[i].dt->get_compute_type();
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (float64)program->fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (float64)program->fetch_result<float64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (float64)program->fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (float64)program->fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (float64)program->fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (float64)program->fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (float64)program->fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (float64)program->fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (float64)program->fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (float64)program->fetch_result<uint64>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

int64 Kernel::get_ret_int(int i) {
  auto dt = rets[i].dt->get_compute_type();
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (int64)program->fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (int64)program->fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (int64)program->fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (int64)program->fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (int64)program->fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (int64)program->fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (int64)program->fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (int64)program->fetch_result<uint64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (int64)program->fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (int64)program->fetch_result<float64>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
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
  } else if (task_type == OffloadedStmt::TaskType::gc) {
    stat.add("launched_tasks_garbage_collect", 1.0);
  }
}

std::string Kernel::get_name() const {
  return name;
}

// static
bool Kernel::supports_lowering(Arch arch) {
  return arch_is_cpu(arch) || (arch == Arch::cuda) || (arch == Arch::metal);
}

TLANG_NAMESPACE_END
