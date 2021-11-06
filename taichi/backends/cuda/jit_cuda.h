#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/jit/jit_session.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/system/timer.h"
#include "taichi/util/file_sequence_writer.h"

#define TI_RUNTIME_HOST
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

TLANG_NAMESPACE_BEGIN

#if defined(TI_WITH_CUDA)
class JITModuleCUDA : public JITModule {
 private:
  void *module_;

 public:
  explicit JITModuleCUDA(void *module) : module_(module) {
  }

  void *lookup_function(const std::string &name) override {
    // TODO: figure out why using the guard leads to wrong tests results
    // auto context_guard = CUDAContext::get_instance().get_guard();
    CUDAContext::get_instance().make_current();
    void *func = nullptr;
    auto t = Time::get_time();
    auto err = CUDADriver::get_instance().module_get_function.call_with_warning(
        &func, module_, name.c_str());
    if (err) {
      TI_ERROR("Cannot look up function {}", name);
    }
    t = Time::get_time() - t;
    TI_TRACE("CUDA module_get_function {} costs {} ms", name, t * 1000);
    TI_ASSERT(func != nullptr);
    return func;
  }

  void call(const std::string &name,
            const std::vector<void *> &arg_pointers) override {
    launch(name, 1, 1, 0, arg_pointers);
  }

  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t dynamic_shared_mem_bytes,
              const std::vector<void *> &arg_pointers) override {
    auto func = lookup_function(name);
    CUDAContext::get_instance().launch(func, name, arg_pointers, grid_dim,
                                       block_dim, dynamic_shared_mem_bytes);
  }

  bool direct_dispatch() const override {
    return false;
  }
};

class JITSessionCUDA : public JITSession {
 public:
  llvm::DataLayout data_layout;

  explicit JITSessionCUDA(llvm::DataLayout data_layout)
      : data_layout(data_layout) {
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override;

  llvm::DataLayout get_data_layout() override {
    return data_layout;
  }

  static std::string compile_module_to_ptx(
      std::unique_ptr<llvm::Module> &module);
};

#endif

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Arch arch);

TLANG_NAMESPACE_END
