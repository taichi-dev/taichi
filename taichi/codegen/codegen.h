// Driver class for kernel code generators.

#pragma once
#include <taichi/runtime/llvm/llvm_runtime_executor.h>
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#ifdef TI_WITH_LLVM
#include "llvm/IR/Module.h"
#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/codegen/llvm/llvm_codegen_utils.h"
#endif
namespace taichi::lang {
class TaichiLLVMContext;

/*
 [Note] Codegen of LLVM-based backends
 * KernelCodeGen is the base class of the codegen of all backends using LLVM.
 * Function `compile_to_function` first compiles the IR of a kernel
 * into a LLVM module using `compile_kernel_to_module`, and then constructs a
 * function for runtime execution using `ModuleToFunctionConverter`.
 *
 * Function `compile_kernel_to_module` compiles the IR of a kernel into a LLVM
 * module. A kernel is composed of several offloaded tasks. To compile a kernel,
 * we first compile each task independently into an LLVM module using function
 * `compile_task`. Then, we link the LLVM modules of the offloaded tasks,
 * the runtime module and the struct modules of the SNode trees which are used
 * in the kernel all together into a single LLVM module using
 * `tlctx->link_compiled_tasks`. The LLVM module and the names of the entry
 * functions of the offloaded tasks in the module are stored in the returned
 * LLVMCompiledKernel.
 *
 * Function `compile_task` uses `TaskCodeGen` of the respective backend to
 * compile the IR of a offloaded task to an LLVM module. It also generates some
 * extra information for linking such as which SNode tree is used in the task.
 * The LLVM module, the name of the entry function of the offloaded task in the
 * module and the extra information are stored in the returned LLVMCompiledTask.
 */
class KernelCodeGen {
 protected:
  Program *prog;
  const Kernel *kernel;
  IRNode *ir;

 public:
  explicit KernelCodeGen(const CompileConfig &compile_config,
                         const Kernel *kernel,
                         IRNode *ir,
                         TaichiLLVMContext &tlctx);

  virtual ~KernelCodeGen() = default;

  static std::unique_ptr<KernelCodeGen> create(
      const CompileConfig &compile_config,
      const Kernel *kernel,
      IRNode *ir,
      TaichiLLVMContext &tlctx);

#ifdef TI_WITH_LLVM
  virtual LLVMCompiledKernel compile_kernel_to_module();

  virtual LLVMCompiledTask compile_task(
      int task_codegen_id,
      const CompileConfig &config,
      std::unique_ptr<llvm::Module> &&module = nullptr,
      IRNode *block = nullptr) {
    TI_NOT_IMPLEMENTED
  }

 protected:
  virtual void optimize_module(llvm::Module *module) {
  }
#endif

  const CompileConfig &get_compile_config() const {
    return compile_config_;
  }

  TaichiLLVMContext &get_taichi_llvm_context() {
    return tlctx_;
  }

 private:
  const CompileConfig &compile_config_;
  TaichiLLVMContext &tlctx_;
};

}  // namespace taichi::lang
