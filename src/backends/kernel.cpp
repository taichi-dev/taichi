#include "kernel.h"
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

FunctionType KernelCodeGen::compile(taichi::Tlang::Program &prog,
                                    taichi::Tlang::Program::Kernel &kernel) {
  //auto t = Time::get_time();
  this->prog = &kernel.program;
  this->kernel = &kernel;
  lower();
  codegen();
  generate_binary("");
  // TC_P(Time::get_time() - t);
  return load_function();
}

TLANG_NAMESPACE_END
