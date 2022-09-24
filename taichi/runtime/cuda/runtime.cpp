#include "taichi/runtime/runtime.h"
#include "taichi/rhi/cuda/cuda_context.h"

namespace taichi::lang {

#if !defined(TI_WITH_CUDA)
static_assert(
    false,
    "This file should not be compiled when TI_WITH_CUDA is undefined");
#endif

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

  ~RuntimeCUDA() override {
  }
};

static class RuntimeCUDAInjector {
 public:
  RuntimeCUDAInjector() {
    Runtime::register_impl<RuntimeCUDA>(Arch::cuda);
  }
} injector;

}  // namespace taichi::lang
