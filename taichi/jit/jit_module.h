#pragma once

#include <memory>
#include <functional>
#include <tuple>

#include "taichi/inc/constants.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/kernel_profiler.h"

namespace taichi::lang {

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

  inline std::tuple<std::vector<void *>, std::vector<int> > get_arg_pointers() {
    return std::make_tuple(std::vector<void *>(), std::vector<int>());
  }

  template <typename... Args, typename T>
  inline std::tuple<std::vector<void *>, std::vector<int> > get_arg_pointers(
      T &t,
      Args &...args) {
    auto [arg_pointers, arg_sizes] = get_arg_pointers(args...);
    arg_pointers.insert(arg_pointers.begin(), &t);
    arg_sizes.insert(arg_sizes.begin(), sizeof(t));
    return std::make_tuple(arg_pointers, arg_sizes);
  }

  // Note: **call** is for serial functions
  // Note: args must pass by value
  // Note: AMDGPU need to pass args by extra_arg currently
  template <typename... Args>
  void call(const std::string &name, Args... args) {
    if (direct_dispatch()) {
      get_function<Args...>(name)(args...);
    } else {
      auto [arg_pointers, arg_sizes] = JITModule::get_arg_pointers(args...);
      call(name, arg_pointers, arg_sizes);
    }
  }

  virtual void call(const std::string &name,
                    const std::vector<void *> &arg_pointers,
                    const std::vector<int> &arg_sizes) {
    TI_NOT_IMPLEMENTED
  }

  // Note: **launch** is for parallel (GPU)_kernels
  // Note: args must pass by value
  template <typename... Args>
  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t shared_mem_bytes,
              Args... args) {
    auto [arg_pointers, arg_sizes] = JITModule::get_arg_pointers(args...);
    launch(name, grid_dim, block_dim, shared_mem_bytes, arg_pointers,
           arg_sizes);
  }

  virtual void launch(const std::string &name,
                      std::size_t grid_dim,
                      std::size_t block_dim,
                      std::size_t shared_mem_bytes,
                      const std::vector<void *> &arg_pointers,
                      const std::vector<int> &arg_sizes) {
    TI_NOT_IMPLEMENTED
  }

  virtual bool direct_dispatch() const = 0;

  virtual ~JITModule() {
  }
};

}  // namespace taichi::lang
