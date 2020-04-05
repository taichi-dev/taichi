#include "taichi/runtime/runtime.h"
#include "taichi/backends/cuda/cuda_context.h"

#include <cuda_runtime.h>

TLANG_NAMESPACE_BEGIN

class RuntimeCUDA : public Runtime {
 public:
  RuntimeCUDA() {
  }

  std::size_t get_total_memory() override {
    return CUDAContext::get_instance().get_total_memory();
  }

  std::size_t get_available_memory() override {
    return CUDAContext::get_instance().get_free_memory();
  }

  bool detected() override {
    return CUDAContext::get_instance().detected();
  }

  ~RuntimeCUDA() {
  }
};

static class RuntimeCUDAInjector {
 public:
  RuntimeCUDAInjector() {
    Runtime::register_impl<RuntimeCUDA>(Arch::cuda);
  }
} injector;

TLANG_NAMESPACE_END
