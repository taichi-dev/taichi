#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/rhi/flagos/flagos_device.h"

namespace taichi {
namespace lang {

// Forward declarations
class KernelCompiler;
class KernelLauncher;

/**
 * @brief FlagOS Program Implementation
 *
 * This class extends LlvmProgramImpl to provide program execution
 * support for the FlagOS unified AI chip backend.
 *
 * Features:
 * - Multi-chip support (MLU, Ascend, DCU, GCU, etc.)
 * - Integration with FlagTree compiler
 * - Optimized kernel compilation and launching
 */

class FlagosProgramImpl : public LlvmProgramImpl {
 public:
  FlagosProgramImpl(CompileConfig &config, KernelProfilerBase *profiler);

  /* ------------------------------------ */
  /* ---- Compilation Interfaces ---- */
  /* ------------------------------------ */

  void compile_snode_tree_types(SNodeTree *tree) override;
  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  /* -------------------------------- */
  /* ---- Runtime Interfaces ---- */
  /* -------------------------------- */

  void materialize_runtime(KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void finalize() override {
    LlvmProgramImpl::finalize();
  }

  void synchronize() override {
    LlvmProgramImpl::synchronize();
  }

  Device *get_compute_device() override {
    return LlvmProgramImpl::get_compute_device();
  }

  /* -------------------------------- */
  /* ---- FlagOS Specific ---- */
  /* -------------------------------- */

  /**
   * @brief Get the target chip for this program
   */
  std::string get_target_chip() const {
    return config_.flagos_chip;
  }

  /**
   * @brief Set the target chip for this program
   */
  void set_target_chip(const std::string &chip) {
    config_.flagos_chip = chip;
  }

  /**
   * @brief Check if FlagOS backend is properly initialized
   */
  bool is_initialized() const;

  ~FlagosProgramImpl() override;

 protected:
  std::unique_ptr<KernelCompiler> make_kernel_compiler() override;
  std::unique_ptr<KernelLauncher> make_kernel_launcher() override;
  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) override;

 private:
  void initialize_flagos_backend();
};

/**
 * @brief Get FlagOS program from Taichi Program
 */
FlagosProgramImpl *get_flagos_program(Program *prog);

}  // namespace lang

}  // namespace taichi
