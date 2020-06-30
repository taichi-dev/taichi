#pragma once

#include "taichi/lang_util.h"
#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCKernel;

class CCKernelGen {
  // Generate corresponding C Source Code for a Taichi Kernel
 public:
  CCKernelGen(Kernel *kernel) : kernel(kernel) {
  }

  std::unique_ptr<CCKernel> compile();

 private:
  Kernel *kernel;
};

FunctionType compile_kernel(Kernel *kernel);

}  // namespace cccp
TLANG_NAMESPACE_END
