#include "kernel.h"

#include "taichi/util/statistics.h"
#include "taichi/common/task.h"
#include "taichi/program/program.h"
#include "taichi/program/async_engine.h"
#include "taichi/codegen/codegen.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/util/action_recorder.h"
#include "taichi/program/extension.h"

TLANG_NAMESPACE_BEGIN

namespace {
class CurrentKernelGuard {
  Kernel *old_kernel;
  Program &program;

 public:
  CurrentKernelGuard(Program &program_, Kernel *kernel) : program(program_) {
    old_kernel = program.current_kernel;
    program.current_kernel = kernel;
  }

  ~CurrentKernelGuard() {
    program.current_kernel = old_kernel;
  }
};
}  // namespace

Kernel::Kernel(Program &program,
               std::function<void()> func,
               std::string primal_name,
               bool grad)
    : program(program), lowered(false), grad(grad) {
  program.initialize_device_llvm_context();
  is_accessor = false;
  is_evaluator = false;
  compiled = nullptr;
  taichi::lang::context = std::make_unique<FrontendContext>();
  ir = taichi::lang::context->get_root();

  {
    CurrentKernelGuard _(program, this);
    program.start_function_definition(this);
    func();
    program.end_function_definition();
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
}

void Kernel::compile() {
  CurrentKernelGuard _(program, this);
  compiled = program.compile(*this);
}

void Kernel::lower(bool to_executable) {  // TODO: is a "Lowerer" class
                                          // necessary for each backend?
  TI_ASSERT(!lowered);
  if (arch_is_cpu(arch) || arch == Arch::cuda || arch == Arch::metal) {
    CurrentKernelGuard _(program, this);
    auto config = program.config;
    bool verbose = config.print_ir;
    if ((is_accessor && !config.print_accessor_ir) ||
        (is_evaluator && !config.print_evaluator_ir))
      verbose = false;

    if (to_executable) {
      irpass::compile_to_executable(
          ir.get(), config, /*vectorize*/ arch_is_cpu(arch), grad,
          /*ad_use_stack=*/true, verbose, /*lower_global_access=*/to_executable,
          /*make_thread_local=*/config.make_thread_local,
          /*make_block_local=*/
          is_extension_supported(config.arch, Extension::bls) &&
              config.make_block_local);
    } else {
      irpass::compile_to_offloads(ir.get(), config, verbose,
                                  /*vectorize=*/arch_is_cpu(arch), grad,
                                  /*ad_use_stack=*/true);
    }
  } else {
    TI_NOT_IMPLEMENTED
  }
  lowered = true;
}

void Kernel::operator()(LaunchContextBuilder &ctx_builder) {
  if (!program.config.async_mode || this->is_evaluator) {
    if (!compiled) {
      compile();
    }

    for (auto &offloaded : ir->as<Block>()->statements) {
      account_for_offloaded(offloaded->as<OffloadedStmt>());
    }

    compiled(ctx_builder.get_context());

    program.sync = (program.sync && arch_is_cpu(arch));
    // Note that Kernel::arch may be different from program.config.arch
    if (program.config.debug && (arch_is_cpu(program.config.arch) ||
                                 program.config.arch == Arch::cuda)) {
      program.check_runtime_error();
    }
  } else {
    program.sync = false;
    program.async_engine->launch(this, ctx_builder.get_context());
    // Note that Kernel::arch may be different from program.config.arch
    if (program.config.debug && arch_is_cpu(arch) &&
        arch_is_cpu(program.config.arch)) {
      program.check_runtime_error();
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

void Kernel::LaunchContextBuilder::set_arg_float(int i, float64 d) {
  TI_ASSERT_INFO(
      !kernel_->args[i].is_nparray,
      "Assigning a scalar value to a numpy array argument is not allowed");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_float64", {ActionArg("kernel_name", kernel_->name),
                                 ActionArg("arg_id", i), ActionArg("val", d)});

  auto dt = kernel_->args[i].dt;
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    ctx_->set_arg(i, (float32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    ctx_->set_arg(i, (float64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    ctx_->set_arg(i, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    ctx_->set_arg(i, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    ctx_->set_arg(i, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    ctx_->set_arg(i, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    ctx_->set_arg(i, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    ctx_->set_arg(i, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    ctx_->set_arg(i, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    ctx_->set_arg(i, (uint64)d);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::LaunchContextBuilder::set_arg_int(int i, int64 d) {
  TI_ASSERT_INFO(
      !kernel_->args[i].is_nparray,
      "Assigning scalar value to numpy array argument is not allowed");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_int64", {ActionArg("kernel_name", kernel_->name),
                               ActionArg("arg_id", i), ActionArg("val", d)});

  auto dt = kernel_->args[i].dt;
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    ctx_->set_arg(i, (int32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    ctx_->set_arg(i, (int64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    ctx_->set_arg(i, (int8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    ctx_->set_arg(i, (int16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    ctx_->set_arg(i, (uint8)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    ctx_->set_arg(i, (uint16)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    ctx_->set_arg(i, (uint32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    ctx_->set_arg(i, (uint64)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    ctx_->set_arg(i, (float32)d);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    ctx_->set_arg(i, (float64)d);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::LaunchContextBuilder::set_extra_arg_int(int i, int j, int32 d) {
  ctx_->extra_args[i][j] = d;
}

void Kernel::LaunchContextBuilder::set_arg_nparray(int i,
                                                   uint64 ptr,
                                                   uint64 size) {
  TI_ASSERT_INFO(kernel_->args[i].is_nparray,
                 "Assigning numpy array to scalar argument is not allowed");

  ActionRecorder::get_instance().record(
      "set_kernel_arg_ext_ptr",
      {ActionArg("kernel_name", kernel_->name), ActionArg("arg_id", i),
       ActionArg("address", fmt::format("0x{:x}", ptr)),
       ActionArg("array_size_in_bytes", (int64)size)});

  kernel_->args[i].size = size;
  ctx_->set_arg(i, ptr);
}

void Kernel::LaunchContextBuilder::set_arg_raw(int i, uint64 d) {
  TI_ASSERT_INFO(
      !kernel_->args[i].is_nparray,
      "Assigning scalar value to numpy array argument is not allowed");

  ActionRecorder::get_instance().record(
      "set_arg_raw", {ActionArg("kernel_name", kernel_->name),
                      ActionArg("arg_id", i), ActionArg("val", (int64)d)});
  ctx_->set_arg<uint64>(i, d);
}

Context &Kernel::LaunchContextBuilder::get_context() {
  ctx_->runtime = static_cast<LLVMRuntime *>(kernel_->program.llvm_runtime);
  return *ctx_;
}

float64 Kernel::get_ret_float(int i) {
  auto dt = rets[i].dt;
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (float64)get_current_program().fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (float64)get_current_program().fetch_result<float64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (float64)get_current_program().fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (float64)get_current_program().fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (float64)get_current_program().fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (float64)get_current_program().fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (float64)get_current_program().fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (float64)get_current_program().fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (float64)get_current_program().fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (float64)get_current_program().fetch_result<uint64>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

int64 Kernel::get_ret_int(int i) {
  auto dt = rets[i].dt;
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return (int64)get_current_program().fetch_result<int32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return (int64)get_current_program().fetch_result<int64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return (int64)get_current_program().fetch_result<int8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return (int64)get_current_program().fetch_result<int16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return (int64)get_current_program().fetch_result<uint8>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return (int64)get_current_program().fetch_result<uint16>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return (int64)get_current_program().fetch_result<uint32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return (int64)get_current_program().fetch_result<uint64>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return (int64)get_current_program().fetch_result<float32>(i);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return (int64)get_current_program().fetch_result<float64>(i);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::set_arch(Arch arch) {
  TI_ASSERT(!compiled);
  this->arch = arch;
}

int Kernel::insert_arg(DataType dt, bool is_nparray) {
  args.push_back(Arg{dt, is_nparray, /*size=*/0});
  return args.size() - 1;
}

int Kernel::insert_ret(DataType dt) {
  rets.push_back(Ret{dt});
  return rets.size() - 1;
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

TLANG_NAMESPACE_END
