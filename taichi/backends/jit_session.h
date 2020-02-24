#include <memory>
#include <functional>
#include "../llvm_fwd.h"
#include "../tlang_util.h"

TLANG_NAMESPACE_BEGIN

// A architecture-specific JIT module that initializes with an **LLVM** module and
// allows the user to call its functions

class JITModule {
 public:
  JITModule(CompileConfig config, std::unique_ptr<llvm::Module> &&module) {
  }

  // Lookup a serial function.
  // For example, a CPU function, or a serial GPU function
  // This function returns a function pointer
  void *lookup_function(const std::string &name) {
    TI_NOT_IMPLEMENTED
  }

  // Lookup a parallel GPU kernel
  // The only argument to GPU kernels should be a Context
  std::function<void()> lookup_spmd_function(const std::string &name) {
    TI_NOT_IMPLEMENTED
  }
};

// Backend JIT environment/compiler for all archs

class JITSession {
 protected:
  std::vector<std::unique_ptr<JITModule>> modules;

 public:
  JITSession() {
  }

  virtual void add_module(std::unique_ptr<llvm::Module> &&module) {
    TI_NOT_IMPLEMENTED
  }

  virtual void call(const std::string &name) {
    TI_NOT_IMPLEMENTED
  }
};

TLANG_NAMESPACE_END
