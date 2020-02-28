#pragma once

#include <memory>
#include <functional>
#include "../llvm_fwd.h"
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

// A architecture-specific JIT module that initializes with an **LLVM** module
// and allows the user to call its functions
// TODO: should we generalize this to include the Metal and OpenGL backends as well?

class JITModule {
 public:
  JITModule() {
  }

  // Lookup a serial function.
  // For example, a CPU function, or a serial GPU function
  // This function returns a function pointer
  virtual void *lookup_function(const std::string &name) = 0;

  // Unfortunately, this can't be virtual since it's a template function
  template <typename... Args>
  std::function<void(Args...)> get_function(const std::string &name) {
    using FuncT = typename std::function<void(Args...)>;
    auto ret = FuncT((function_pointer_type<FuncT>)lookup_function(name));
    TI_ASSERT(ret != nullptr);
    return ret;
  }

  template <typename... Args>
  void call(const std::string &name, Args... args) {
    get_function<Args...>(name)(args...);
  }

  virtual ~JITModule() {
  }
};

TLANG_NAMESPACE_END
