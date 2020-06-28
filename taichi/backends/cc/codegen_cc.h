#pragma once

#include "taichi/lang_util.h"
#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCKernel;
class CCLayout;
class CCLauncher;

class CCKernelGen {
  // Generate corresponding C Source Code for a Taichi Kernel
 public:
  CCKernelGen(
      Program *program,
      Kernel *kernel,
      CCLayout *layout)
      : program(program)
      , kernel(kernel)
      , layout(layout) {
  }

  std::unique_ptr<CCKernel> compile();

 private:
  Program *program;
  Kernel *kernel;
  CCLayout *layout;
};

FunctionType compile_kernel(
      Program *program,
      Kernel *kernel,
      CCLayout *layout, 
      CCLauncher *launcher);

}  // namespace cccp
TLANG_NAMESPACE_END
