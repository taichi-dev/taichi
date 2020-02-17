#pragma once

#include <taichi/common.h>
#include <taichi/constants.h>
#include <taichi/tlang_util.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base.h"
#include "kernel.h"

TLANG_NAMESPACE_BEGIN
namespace opengl {

class OpenglCodeGen {
 public:
  OpenglCodeGen(const std::string &kernel_name)
    : kernel_name_(kernel_name)
  {}

  FunctionType compile(Program &program, Kernel &kernel);

 private:
  void lower();
  FunctionType gen();

  const std::string kernel_name_;

  Program *prog_;
  Kernel *kernel_;
  size_t global_tmps_buffer_size_{0};
};

} // namespace opengl
TLANG_NAMESPACE_END
