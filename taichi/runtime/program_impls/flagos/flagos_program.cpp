#include "taichi/runtime/program_impls/flagos/flagos_program.h"

#include "taichi/codegen/flagos/codegen_flagos.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/program/program.h"
#include "taichi/runtime/llvm/llvm_aot_module_builder.h"
#include "taichi/runtime/program_impls/flagos/flagos_kernel_compiler.h"
#include "taichi/runtime/program_impls/flagos/flagos_kernel_launcher.h"

namespace taichi {
namespace lang {

FlagosProgramImpl::FlagosProgramImpl(CompileConfig &config,
                                     KernelProfilerBase *profiler)
    : LlvmProgramImpl(config, profiler) {
  // Initialize FlagOS-specific settings
  if (config.flagos_chip.empty()) {
    // Try to get from environment variable
    const char *chip_env = std::getenv("TI_FLAGOS_CHIP");
    if (chip_env) {
      config.flagos_chip = chip_env;
    } else {
      config.flagos_chip = "generic";  // Default chip
    }
  }

  TI_INFO("FlagosProgramImpl created with chip: {}", config.flagos_chip);

  // Validate chip name
  if (!flagos::FlagosDevice::is_chip_supported(config.flagos_chip)) {
    TI_WARN("FlagOS chip '{}' may not be fully supported", config.flagos_chip);
  }
}

FlagosProgramImpl::~FlagosProgramImpl() {
  // Cleanup is handled by base class
}

void FlagosProgramImpl::initialize_flagos_backend() {
  auto *device = static_cast<flagos::FlagosDevice *>(llvm_device());
  if (device) {
    device->initialize(config_.flagos_chip);
    TI_INFO("FlagOS backend initialized for chip: {}", config_.flagos_chip);
  }
}

bool FlagosProgramImpl::is_initialized() const {
  auto *device = static_cast<flagos::FlagosDevice *>(llvm_device());
  if (device) {
    return flagos::FlagosDevice::is_chip_supported(device->get_target_chip());
  }
  return false;
}

void FlagosProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  // Delegate to base class implementation
  LlvmProgramImpl::compile_snode_tree_types(tree);
}

void FlagosProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                               uint64 *result_buffer) {
  // Initialize FlagOS backend on first materialization
  static bool flagos_initialized = false;
  if (!flagos_initialized) {
    initialize_flagos_backend();
    flagos_initialized = true;
  }

  // Delegate to base class implementation
  LlvmProgramImpl::materialize_snode_tree(tree, result_buffer);
}

void FlagosProgramImpl::materialize_runtime(KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  // First initialize base LLVM runtime
  LlvmProgramImpl::materialize_runtime(profiler, result_buffer_ptr);

  // Then initialize FlagOS-specific context
  initialize_flagos_backend();
}

std::unique_ptr<KernelCompiler> FlagosProgramImpl::make_kernel_compiler() {
  flagos::KernelCompiler::Config cfg;
  cfg.tlctx = get_llvm_context();
  return std::make_unique<flagos::KernelCompiler>(std::move(cfg));
}

std::unique_ptr<KernelLauncher> FlagosProgramImpl::make_kernel_launcher() {
  flagos::KernelLauncher::Config cfg;
  cfg.executor = get_runtime_executor();
  return std::make_unique<flagos::KernelLauncher>(std::move(cfg));
}

std::unique_ptr<AotModuleBuilder> FlagosProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  // Use LLVM AOT module builder for now
  // TODO: Create FlagOS-specific AOT module builder
  return std::make_unique<LlvmAotModuleBuilder>(
      get_kernel_compilation_manager(), config_, this);
}

FlagosProgramImpl *get_flagos_program(Program *prog) {
  auto *result = dynamic_cast<FlagosProgramImpl *>(prog->get_program_impl());
  TI_ASSERT(result);
  return result;
}

}  // namespace lang

}  // namespace taichi
