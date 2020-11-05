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

class OpenclKernel {
  std::string name;
  std::string source;

 public:
  OpenclKernel(std::string name, std::string const &source)
    : name(name), source(source) {}

  void launch(Context *ctx) {}
};

}  // namespace opencl
TLANG_NAMESPACE_END

