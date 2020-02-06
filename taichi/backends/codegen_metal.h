#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <taichi/common.h>
#include <taichi/constants.h>
#include <taichi/platform/metal/metal_data_types.h>
#include <taichi/platform/metal/metal_kernel_util.h>
#include <taichi/platform/metal/metal_runtime.h>
#include <taichi/tlang_util.h>
#include "base.h"
#include "kernel.h"

#ifdef TC_SUPPORTS_METAL

TLANG_NAMESPACE_BEGIN
namespace metal {

class MetalCodeGen {
 public:
  MetalCodeGen(const std::string &kernel_name,
               const StructCompiledResult *struct_compiled);

  FunctionType compile(Program &, Kernel &kernel, MetalRuntime *runtime);

 private:
  void lower();
  FunctionType gen(MetalRuntime *runtime);

  const int id_;
  const std::string taichi_kernel_name_;
  const StructCompiledResult *const struct_compiled_;

  Program *prog_;
  Kernel *kernel_;
  size_t global_tmps_buffer_size_{0};
};

}  // namespace metal

TLANG_NAMESPACE_END

#endif  // TC_SUPPORTS_METAL
