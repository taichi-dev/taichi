#include "kernel.h"

#include "taichi/common/task.h"
#include "taichi/program/program.h"
#include "taichi/program/async_engine.h"
#include "taichi/codegen/codegen.h"

#if defined(TI_WITH_CUDA)
#include <cuda_runtime.h>
#include "taichi/backends/cuda/cuda_utils.h"
#endif

TLANG_NAMESPACE_BEGIN

Kernel::Kernel(Program &program,
               std::function<void()> func,
               std::string primal_name,
               bool grad)
    : program(program), lowered(false), grad(grad) {
  program.initialize_device_llvm_context();
  is_accessor = false;
  compiled = nullptr;
  taichi::lang::context = std::make_unique<FrontendContext>();
  ir_holder = taichi::lang::context->get_root();
  ir = ir_holder.get();

  program.current_kernel = this;
  program.start_function_definition(this);
  func();
  program.end_function_definition();
  program.current_kernel = nullptr;

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
  program.current_kernel = this;
  compiled = program.compile(*this);
  program.current_kernel = nullptr;
}

void Kernel::lower() {  // TODO: is a "Lowerer" class necessary for each
                        // backend?
  TI_ASSERT(!lowered);
  if (arch_is_cpu(arch) || arch == Arch::cuda) {
    auto codegen = KernelCodeGen::create(arch, this);
    auto config = program.config;
    bool verbose = config.print_ir;
    if (is_accessor && !config.print_accessor_ir)
      verbose = false;
    irpass::compile_to_offloads(ir, config, /*vectorize*/ arch_is_cpu(arch),
                                grad,
                                /*ad_use_stack*/ true, verbose);
  } else {
    TI_NOT_IMPLEMENTED
  }
  lowered = true;
}

void Kernel::operator()() {
  if (!program.config.async) {
    if (!compiled) {
      compile();
    }
    compiled(program.get_context());
    program.sync = (program.sync && arch_is_cpu(arch));
    if (program.config.debug && arch_is_cpu(arch)) {
      program.check_runtime_error();
    }
  } else {
    program.engine->launch(this);
  }
}

void Kernel::set_arg_float(int i, float64 d) {
  TI_ASSERT_INFO(
      !args[i].is_nparray,
      "Assigning a scalar value to a numpy array argument is not allowed");
  auto dt = args[i].dt;
  if (dt == DataType::f32) {
    program.context.set_arg(i, (float32)d);
  } else if (dt == DataType::f64) {
    program.context.set_arg(i, (float64)d);
  } else if (dt == DataType::i32) {
    program.context.set_arg(i, (int32)d);
  } else if (dt == DataType::i64) {
    program.context.set_arg(i, (int64)d);
  } else if (dt == DataType::i8) {
    program.context.set_arg(i, (int8)d);
  } else if (dt == DataType::i16) {
    program.context.set_arg(i, (int16)d);
  } else if (dt == DataType::u8) {
    program.context.set_arg(i, (uint8)d);
  } else if (dt == DataType::u16) {
    program.context.set_arg(i, (uint16)d);
  } else if (dt == DataType::u32) {
    program.context.set_arg(i, (uint32)d);
  } else if (dt == DataType::u64) {
    program.context.set_arg(i, (uint64)d);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::set_extra_arg_int(int i, int j, int32 d) {
  program.context.extra_args[i][j] = d;
}

void Kernel::set_arg_int(int i, int64 d) {
  TI_ASSERT_INFO(
      args[i].is_nparray == false,
      "Assigning scalar value to numpy array argument is not allowed");
  auto dt = args[i].dt;
  if (dt == DataType::i32) {
    program.context.set_arg(i, (int32)d);
  } else if (dt == DataType::i64) {
    program.context.set_arg(i, (int64)d);
  } else if (dt == DataType::i8) {
    program.context.set_arg(i, (int8)d);
  } else if (dt == DataType::i16) {
    program.context.set_arg(i, (int16)d);
  } else if (dt == DataType::u8) {
    program.context.set_arg(i, (uint8)d);
  } else if (dt == DataType::u16) {
    program.context.set_arg(i, (uint16)d);
  } else if (dt == DataType::u32) {
    program.context.set_arg(i, (uint32)d);
  } else if (dt == DataType::u64) {
    program.context.set_arg(i, (uint64)d);
  } else if (dt == DataType::f32) {
    program.context.set_arg(i, (float32)d);
  } else if (dt == DataType::f64) {
    program.context.set_arg(i, (float64)d);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void Kernel::mark_arg_return_value(int i, bool is_return) {
  args[i].is_return_value = is_return;
}

void Kernel::set_arg_nparray(int i, uint64 ptr, uint64 size) {
  TI_ASSERT_INFO(args[i].is_nparray,
                 "Assigning numpy array to scalar argument is not allowed");
  args[i].size = size;
  program.context.set_arg(i, ptr);
}

void Kernel::set_arch(Arch arch) {
  TI_ASSERT(!compiled);
  this->arch = arch;
}

int Kernel::insert_arg(DataType dt, bool is_nparray) {
  args.push_back(Arg{dt, is_nparray, 0, false});
  return args.size() - 1;
}

TLANG_NAMESPACE_END
