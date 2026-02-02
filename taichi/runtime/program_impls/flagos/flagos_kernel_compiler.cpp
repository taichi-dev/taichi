#include "taichi/runtime/program_impls/flagos/flagos_kernel_compiler.h"

#include "taichi/program/kernel.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi {
namespace lang {
namespace flagos {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

LLVM::CompiledKernelData KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel,
    IRNode *ir) {
  return compile_with_flagtree(compile_config, kernel, ir);
}

LLVM::CompiledKernelData KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel,
    IRNode *ir,
    const LLVM::CompiledKernelData &cached_data) {
  // Check if cached data is compatible
  if (cached_data.arch() == compile_config.arch) {
    TI_TRACE("Using cached FlagOS kernel: {}", kernel.get_name());
    return cached_data;
  }

  return compile_with_flagtree(compile_config, kernel, ir);
}

LLVM::CompiledKernelData KernelCompiler::compile_with_flagtree(
    const CompileConfig &compile_config,
    const Kernel &kernel,
    IRNode *ir) {
  TI_TRACE("Compiling FlagOS kernel: {}", kernel.get_name());

  // Create FlagOS code generator
  auto codegen =
      KernelCodeGenFlagOS(compile_config, &kernel, ir, *config_.tlctx);

  // Compile to LLVM IR
  auto compiled_tasks = codegen.run_compilation();

  // Create compiled kernel data
  LLVM::CompiledKernelData::InternalData data;
  data.arch = compile_config.arch;
  data.kernel_key = get_cache_key(&kernel);
  data.tasks = std::move(compiled_tasks.tasks);
  data.args = std::move(compiled_tasks.args);
  data.module = std::move(compiled_tasks.module);

  // TODO: Use FlagTree to compile LLVM IR to target chip code
  // auto flagtree = FlagTreeCompiler();
  // auto chip_code = flagtree.compile(data.module, compile_config.flagos_chip);
  // data.chip_binary = std::move(chip_code);

  return LLVM::CompiledKernelData(std::move(data));
}

}  // namespace flagos
}  // namespace lang
}  // namespace taichi
