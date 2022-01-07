#pragma once

#include <memory>
#include <functional>

#include "taichi/llvm/llvm_fwd.h"
#include "taichi/lang_util.h"
#include "taichi/jit/jit_module.h"

TLANG_NAMESPACE_BEGIN

// Backend JIT compiler for all archs

class Program;

class JITSession {
 protected:
  std::vector<std::unique_ptr<JITModule>> modules;
  Program *prog;

 public:
  JITSession(Program *program);

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M,
                                int max_reg = 0) = 0;

  // virtual void remove_module(JITModule *module) = 0;

  virtual void *lookup(const std::string Name) {
    TI_NOT_IMPLEMENTED
  }

  virtual llvm::DataLayout get_data_layout();

  static std::unique_ptr<JITSession> create(Program *prog, Arch arch);

  virtual void global_optimize_module(llvm::Module *module) {
  }

  virtual ~JITSession() = default;
};

TLANG_NAMESPACE_END
