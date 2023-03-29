#pragma once

#include <memory>
#include <functional>

#include "taichi/runtime/llvm/llvm_fwd.h"
#include "taichi/util/lang_util.h"
#include "taichi/jit/jit_module.h"

namespace taichi::lang {

// Backend JIT compiler for all archs

class TaichiLLVMContext;
struct CompileConfig;

class JITSession {
 protected:
  TaichiLLVMContext *tlctx_;
  const CompileConfig &config_;

  std::vector<std::unique_ptr<JITModule>> modules;

 public:
  JITSession(TaichiLLVMContext *tlctx, const CompileConfig &config);

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M,
                                int max_reg = 0) = 0;

  // virtual void remove_module(JITModule *module) = 0;

  virtual void *lookup(const std::string Name) {
    TI_NOT_IMPLEMENTED
  }

  virtual llvm::DataLayout get_data_layout() = 0;

  static std::unique_ptr<JITSession> create(TaichiLLVMContext *tlctx,
                                            const CompileConfig &config,
                                            Arch arch);

  virtual ~JITSession() = default;
};

}  // namespace taichi::lang
