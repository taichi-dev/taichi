#pragma once

#include "taichi/lang_util.h"
#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN

class CCProgramImpl;

namespace cccp {

class CCKernel;

class CCKernelGen {
  // Generate corresponding C Source Code for a Taichi Kernel
 public:
  CCKernelGen(Kernel *kernel, CCProgramImpl *cc_program_impl)
      : cc_program_impl_(cc_program_impl), kernel_(kernel) {
  }

  std::unique_ptr<CCKernel> compile();

 private:
  CCProgramImpl *cc_program_impl_{nullptr};
  Kernel *kernel_;
};

FunctionType compile_kernel(Kernel *kernel);

}  // namespace cccp
TLANG_NAMESPACE_END
