#include <cuda_runtime.h>
#include <taichi/common/task.h>
#include <taichi/taichi>
#include "program.h"
#include "snode.h"
#include "backends/struct.h"
#include "backends/cpu.h"
#include "backends/gpu.h"

TLANG_NAMESPACE_BEGIN

Program *current_program = nullptr;
SNode root;

Program::Kernel::Kernel(Program &program,
                        std::function<void()> func,
                        std::string name)
    : program(program), name(name) {
  compiled = nullptr;
  benchmarking = false;
  context = std::make_unique<FrontendContext>();
  ir_holder = context->get_root();
  ir = ir_holder.get();

  program.current_kernel = this;
  program.start_function_definition(this);
  func();
  program.end_function_definition();
  program.current_kernel = nullptr;
}

void Program::Kernel::compile() {
  program.current_kernel = this;
  compiled = program.compile(*this);
  program.current_kernel = nullptr;
}

void Program::Kernel::operator()() {
  if (!compiled)
    compile();
  auto c = program.get_context();
  auto t = Time::get_time();
  compiled(c);
  TC_P((Time::get_time() - t) * 1000);

  program.sync = false;
}

FunctionType Program::compile(Kernel &kernel) {
  FunctionType ret = nullptr;
  if (config.arch == Arch::x86_64) {
    CPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else if (config.arch == Arch::gpu) {
    GPUCodeGen codegen(kernel.name);
    ret = codegen.compile(*this, kernel);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  return ret;
}

void Program::materialize_layout() {
  StructCompiler scomp;
  scomp.run(root);
  layout_fn = scomp.get_source_path();
  data_structure = scomp.creator();
}

void Program::synchronize() {
  if (!sync) {
    if (config.arch == Arch::gpu) {
      cudaDeviceSynchronize();
    }
    sync = true;
  }
}

TLANG_NAMESPACE_END
