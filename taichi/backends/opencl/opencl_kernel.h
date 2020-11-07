#pragma once

#include "taichi/lang_util.h"
#include <vector>
#include <memory>

TLANG_NAMESPACE_BEGIN

struct Context;

namespace opencl {

class OpenclProgram;

class OpenclKernel {
  std::string name;
  std::string source;

 public:
  struct Impl;
  std::unique_ptr<Impl> impl;

  OpenclKernel(OpenclProgram *prog, Kernel *kernel,
      int offload_count, std::string const &source);
  ~OpenclKernel();

  void launch(Context *ctx);
};

}  // namespace opencl
TLANG_NAMESPACE_END
