#pragma once

#include "taichi/codegen/llvm/codegen_llvm.h"

namespace taichi {
namespace lang {

/**
 * @brief LLVM Code Generation for FlagOS Backend
 *
 * This class extends TaskCodeGenLLVM to provide code generation
 * for the FlagOS unified AI chip backend. It generates LLVM IR
 * that can be compiled by FlagTree to target various AI chips.
 */

class TaskCodeGenFlagOS : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenFlagOS(int id,
                    const CompileConfig &config,
                    TaichiLLVMContext &tlctx,
                    const Kernel *kernel,
                    IRNode *ir = nullptr)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir) {
  }

  // Custom visitors for FlagOS-specific code generation
  void visit(OffloadedStmt *stmt) override;
  void visit(PrintStmt *stmt) override;
  void visit(ExternalFuncCallStmt *stmt) override;

  // Parallel for constructs
  void create_offload_range_for(OffloadedStmt *stmt) override;
  void create_offload_struct_for(OffloadedStmt *stmt) override;
  void create_offload_mesh_for(OffloadedStmt *stmt) override;

 protected:
  // FlagOS-specific code generation helpers
  void emit_flagos_gc(OffloadedStmt *stmt);
  void create_bls_buffer(OffloadedStmt *stmt) override;

  // SPMD (Single Program Multiple Data) information
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override;

  // Target-specific optimizations
  llvm::Value *optimized_reduction(AtomicOpStmt *stmt) override;

  // Kernel argument passing
  bool kernel_argument_by_val() const override {
    return false;
  }

  // Math function emission using FlagOS math library
  void emit_extra_unary(UnaryOpStmt *stmt) override;
  void visit(BinaryOpStmt *stmt) override;

  // Print support
  void create_print(std::string tag, DataType dt, llvm::Value *value) override;
  std::tuple<llvm::Value *, llvm::Type *> create_value_and_type(
      llvm::Value *value,
      DataType dt);
};

/**
 * @brief Kernel Code Generation for FlagOS
 */
class KernelCodeGenFlagOS : public KernelCodeGen {
 public:
  KernelCodeGenFlagOS(const CompileConfig &compile_config,
                      const Kernel *kernel,
                      IRNode *ir = nullptr,
                      TaichiLLVMContext &tlctx)
      : KernelCodeGen(compile_config, kernel, ir, tlctx) {
  }

  LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) override;
};

}  // namespace lang

}  // namespace taichi
