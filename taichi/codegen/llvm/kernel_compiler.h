#pragma once

#include "taichi/codegen/kernel_compiler.h"
#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi::lang {
namespace LLVM {

class KernelCompiler : public lang::KernelCompiler {
 public:
  struct Config {
    TaichiLLVMContext *tlctx{nullptr};
  };

  explicit KernelCompiler(Config config);

  IRNodePtr compile(const CompileConfig &compile_config,
                    const Kernel &kernel_def) const override;

  CKDPtr compile(const CompileConfig &compile_config,
                 const DeviceCapabilityConfig &device_caps,
                 const Kernel &kernel_def,
                 IRNode &chi_ir) const override;

 private:
  Config config_;
};

}  // namespace LLVM
}  // namespace taichi::lang
