#pragma once

#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

class CCKernel;

class CCLauncher {
  // Launch C compiler to compile generated source code, and run them
 public:
  CCLauncher();
  ~CCLauncher();

  void launch(CCKernel *kernel, Context *ctx);
  void keep(std::unique_ptr<CCKernel> kernel);

 private:
  std::vector<std::unique_ptr<CCKernel>> kept_kernels;
};

}  // namespace cccp
TLANG_NAMESPACE_END
