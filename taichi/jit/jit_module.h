#pragma once

#include <memory>
#include <functional>

#include "taichi/inc/constants.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/lang_util.h"
#include "taichi/program/profiler.h"

TLANG_NAMESPACE_BEGIN

// A architecture-specific JIT module that initializes with an **LLVM** module
// and allows the user to call its functions
// TODO: should we generalize this to include the Metal and OpenGL backends as
// well?

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

  static std::vector<void *> get_arg_pointers() {
    return std::vector<void *>();
  }

  template <typename... Args, typename T>
  static std::vector<void *> get_arg_pointers(T &t, Args &... args) {
    auto ret = get_arg_pointers(args...);
    ret.insert(ret.begin(), &t);
    return ret;
  }

  // Note: **call** is for serial functions
  // Note: args must pass by value
  template <typename... Args>
  void call(const std::string &name, Args... args) {
    if (direct_dispatch()) {
      get_function<Args...>(name)(args...);
    } else {
      auto arg_pointers = JITModule::get_arg_pointers(args...);
      call(name, arg_pointers);
    }
  }

  virtual void call(const std::string &name,
                    const std::vector<void *> &arg_pointers) {
    TI_NOT_IMPLEMENTED
  }

  // Note: **launch** is for parallel (GPU)_kernels
  // Note: args must pass by value
  template <typename... Args>
  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              Args... args) {
    auto arg_pointers = JITModule::get_arg_pointers(args...);
    launch_with_arg_pointers(name, grid_dim, block_dim, arg_pointers);
  }

  virtual void launch(const std::string &name,
                      std::size_t grid_dim,
                      std::size_t block_dim,
                      const std::vector<void *> &arg_pointers) {
    TI_NOT_IMPLEMENTED
  }

  // directly call the function (e.g. on CPU), or via another runtime system
  // (e.g. cudaLaunch)?
  virtual bool direct_dispatch() const = 0;

  virtual uint64 fetch_result_u64() = 0;

  template <typename T>
  T fetch_result() {
    static_assert(sizeof(T) <= sizeof(uint64));
    return taichi_union_cast<T>(fetch_result_u64());
  }

  virtual ~JITModule() {
  }
};

TLANG_NAMESPACE_END
