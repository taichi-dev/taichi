#pragma once

#include <memory>
#include <functional>

#include "taichi/inc/constants.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/kernel_profiler.h"
#include "taichi/rhi/arch.h"

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

  inline std::vector<void *> get_arg_pointers() {
    return std::vector<void *>();
  }

  template <typename... Args, typename T>
  inline std::vector<void *> get_arg_pointers(T &t, Args &...args) {
    auto ret = get_arg_pointers(args...);
    ret.insert(ret.begin(), &t);
    return ret;
  }

  inline int get_args_bytes() {
    return 0;
  }

  template <typename... Args, typename T>
  inline int get_args_bytes(T t, Args ...args) {
    return get_args_bytes(args...) + sizeof(T); 
  }

  inline void init_args_pointers(char *packed_args) {
    return ;
  }

  template <typename... Args, typename T>
  inline void init_args_pointers(char *packed_args, T t, Args ...args) {
      std::memcpy(packed_args, &t, sizeof(t));
      init_args_pointers(packed_args + sizeof(t), args...);
      return ;
  }

  // Note: **call** is for serial functions
  // Note: args must pass by value
  // Note: AMDGPU need to pass args by extra_arg currently
  template <typename... Args>
  void call(const std::string &name, Args... args) {
    if (direct_dispatch()) {
      get_function<Args...>(name)(args...);
    } else {
      if (module_arch() == Arch::cuda) {
#if defined(TI_WITH_CUDA)
        auto arg_pointers = JITModule::get_arg_pointers(args...);
        call(name, arg_pointers);
#else
  TI_NOT_IMPLEMENTED
#endif
      }
      else if (module_arch() == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
        auto arg_bytes = JITModule::get_args_bytes(args...);
        char packed_args[arg_bytes];
        JITModule::init_args_pointers(packed_args, args...);
        call(name, { (void*)packed_args , (void*)&arg_bytes});
#else
  TI_NOT_IMPLEMENTED
#endif
      }
      else {
        TI_ERROR("unknown module arch")
      }
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
              std::size_t shared_mem_bytes,
              Args... args) {
    auto arg_pointers = JITModule::get_arg_pointers(args...);
    launch(name, grid_dim, block_dim, shared_mem_bytes, arg_pointers);
  }

  virtual void launch(const std::string &name,
                      std::size_t grid_dim,
                      std::size_t block_dim,
                      std::size_t shared_mem_bytes,
                      const std::vector<void *> &arg_pointers) {
    TI_NOT_IMPLEMENTED
  }

  virtual bool direct_dispatch() const = 0;
  virtual Arch module_arch() const = 0;

  virtual ~JITModule() {
  }
};

}  // namespace taichi::lang
