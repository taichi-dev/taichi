#pragma once

#include <memory>

#include "taichi/rhi/arch.h"
#include "taichi/program/kernel.h"
#include "taichi/program/compile_config.h"
#include "taichi/codegen/compiled_kernel_data.h"

namespace taichi::lang {

class KernelCompiler {
 public:
  virtual std::unique_ptr<IRNode> compile(const CompileConfig &compile_config,
                                          const Kernel &kernel_def) const = 0;

  virtual std::unique_ptr<CompiledKernelData> compile(
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &device_caps,
      const Kernel &kernel_def,
      const IRNode &chi_ir) const = 0;

  virtual ~KernelCompiler() = default;
};

}  // namespace taichi::lang
