#pragma once

#include "taichi/lang_util.h"
#include <vector>
#include <memory>

TLANG_NAMESPACE_BEGIN

class SNode;
struct Context;
class Program;
class Kernel;

namespace opencl {

class OpenclKernel;

class OpenclProgram {
  Program *prog;
  std::string layout_source;

  std::vector<std::unique_ptr<OpenclKernel>> kernels;

 public:
  struct Impl;
  std::unique_ptr<Impl> impl;

  static bool is_opencl_api_available();

  OpenclProgram(Program *prog);
  ~OpenclProgram();

  FunctionType compile_kernel(Kernel *kernel);
  void compile_layout(SNode *root);
  std::string get_header_lines();
};

}  // namespace opencl
TLANG_NAMESPACE_END
