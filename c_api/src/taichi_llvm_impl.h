#pragma once
#ifdef TI_WITH_LLVM

#include "taichi_core_impl.h"

#ifdef TI_WITH_CUDA
#include "taichi/platform/cuda/detect_cuda.h"
#endif

namespace taichi::lang {
class LlvmRuntimeExecutor;
struct CompileConfig;
}  // namespace taichi::lang

namespace capi {

class LlvmRuntime : public Runtime {
 public:
  LlvmRuntime(taichi::Arch arch);
  virtual ~LlvmRuntime();

  void check_runtime_error();
  taichi::lang::Device &get() override;

 private:
  /* Internally used interfaces */
  TiAotModule load_aot_module(const char *module_path) override;
  TiMemory allocate_memory(
      const taichi::lang::Device::AllocParams &params) override;
  void free_memory(TiMemory devmem) override;

  void buffer_copy(const taichi::lang::DevicePtr &dst,
                   const taichi::lang::DevicePtr &src,
                   size_t size) override;

  void flush() override;

  void wait() override;

 private:
  std::unique_ptr<taichi::lang::CompileConfig> cfg_{nullptr};
  std::unique_ptr<taichi::lang::LlvmRuntimeExecutor> executor_{nullptr};
  taichi::uint64 *result_buffer{nullptr};
};

}  // namespace capi

#endif  // TI_WITH_LLVM
