#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCKernel;
class CCLayout;

class CCProgram {
  // Launch C compiler to compile generated source code, and run them
 public:
  CCProgram();
  ~CCProgram();

  void launch(CCKernel *kernel, Context *ctx);

  std::vector<std::unique_ptr<CCKernel>> kernels;
  std::unique_ptr<CCLayout> layout;
};

}  // namespace cccp
TLANG_NAMESPACE_END
