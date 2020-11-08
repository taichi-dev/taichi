#pragma once

#include "taichi/lang_util.h"
#include <vector>
#include <memory>

TLANG_NAMESPACE_BEGIN

struct Context;

namespace opencl {

class OpenclProgram;

struct OpenclOffloadMeta {
  std::string kernel_name;
  int grid_dim{1};
  int block_dim{1};
};

class OpenclKernel {
  std::string name;
  std::string source;

 public:
  struct Impl;
  std::unique_ptr<Impl> impl;

  OpenclKernel(OpenclProgram *prog, Kernel *kernel,
      std::vector<OpenclOffloadMeta> const &offloads,
      std::string const &source);
  ~OpenclKernel();

  void launch(Context *ctx);
};

}  // namespace opencl
TLANG_NAMESPACE_END
