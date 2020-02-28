#pragma once

#include <taichi/inc/constants.h>
#include <taichi/platform/metal/metal_data_types.h>
#include <taichi/platform/metal/metal_kernel_util.h>
#include <taichi/platform/metal/metal_runtime.h>
#include <taichi/lang_util.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "codegen.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

class MetalCodeGen {
 public:
  MetalCodeGen(const std::string &kernel_name,
               const StructCompiledResult *struct_compiled);

  FunctionType compile(Program &, Kernel &kernel, MetalRuntime *runtime);

 private:
  void lower();
  FunctionType gen(const SNode &root_snode, MetalRuntime *runtime);

  const int id_;
  const std::string taichi_kernel_name_;
  const StructCompiledResult *const struct_compiled_;

  Program *prog_;
  Kernel *kernel_;
  size_t global_tmps_buffer_size_{0};
};

}  // namespace metal

TLANG_NAMESPACE_END
