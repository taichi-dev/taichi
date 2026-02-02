#pragma once

#include "taichi/codegen/llvm/kernel_compiler.h"
#include "taichi/codegen/flagos/codegen_flagos.h"

namespace taichi {
namespace lang {
namespace flagos {

/**
 * @brief FlagOS Kernel Compiler
 *
 * Compiles Taichi kernels to LLVM IR for FlagOS backend.
 * Uses FlagTree to compile LLVM IR to target chip code.
 */

class KernelCompiler : public LLVM::KernelCompiler {
 public:
  using IRNode = taichi::lang::IRNode;
  using Config = LLVM::KernelCompiler::Config;

  explicit KernelCompiler(Config config);

  /**
   * @brief Compile a kernel to FlagOS executable format
   */
  LLVM::CompiledKernelData compile(const CompileConfig &compile_config,
                                   const DeviceCapabilityConfig &device_caps,
                                   const Kernel &kernel,
                                   IRNode *ir) override;

  /**
   * @brief Compile a kernel with cached data
   */
  LLVM::CompiledKernelData compile(
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &device_caps,
      const Kernel &kernel,
      IRNode *ir,
      const LLVM::CompiledKernelData &cached_data) override;

 private:
  Config config_;

  /**
   * @brief Compile kernel using FlagTree
   */
  LLVM::CompiledKernelData compile_with_flagtree(
      const CompileConfig &compile_config,
      const Kernel &kernel,
      IRNode *ir);
};

}  // namespace flagos
}  // namespace lang
}  // namespace taichi
