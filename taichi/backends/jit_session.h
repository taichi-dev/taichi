#pragma once

#include <memory>
#include <functional>
#include "../llvm_fwd.h"
#include "../tlang_util.h"

TLANG_NAMESPACE_BEGIN

// A architecture-specific JIT module that initializes with an **LLVM** module
// and allows the user to call its functions

class JITModule {
 public:
  JITModule() {
  }

  // Lookup a serial function.
  // For example, a CPU function, or a serial GPU function
  // This function returns a function pointer
  virtual void *lookup_function(const std::string &name) {
    TI_NOT_IMPLEMENTED
  }

  // Unfortunately, this can't be virtual since it's a template function
  template <typename... Args>
  std::function<void(Args...)> get_function(const std::string &name) {
    using FuncT = typename std::function<void(Args...)>;
    auto ret = FuncT((function_pointer_type<FuncT>)lookup_function(name));
    TI_ASSERT(ret != nullptr);
    return ret;
  }

  template <typename... Args>
  void call(const std::string &name, Args &&... args) {
    get_function<Args...>(name)(std::forward<Args>(args)...);
  }

  // Lookup a parallel GPU kernel
  // The only argument to GPU kernels should be a Context
  virtual std::function<void()> lookup_spmd_function(const std::string &name) {
    TI_NOT_IMPLEMENTED
  }

  virtual ~JITModule() {
  }
};

// Backend JIT compiler for all archs

class JITSession {
 protected:
  std::vector<std::unique_ptr<JITModule>> modules;

 public:
  JITSession() {
  }

  virtual const llvm::DataLayout &get_data_layout() const = 0;

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M) = 0;

  // virtual void remove_module(JITModule *module) = 0;

  virtual void *lookup(const std::string Name) = 0;

  virtual std::size_t get_type_size(llvm::Type *type) const = 0;

  virtual ~JITSession() = default;
};

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Arch arch);

TLANG_NAMESPACE_END
