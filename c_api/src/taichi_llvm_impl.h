#pragma once

#include "taichi_core_impl.h"

namespace taichi {
namespace lang {
class LlvmRuntimeExecutor;
class MemoryPool;
struct CompileConfig;
}  // namespace lang
}  // namespace taichi

namespace capi {

class LlvmRuntime : public Runtime {
 public:
  LlvmRuntime(taichi::Arch arch);

  void check_runtime_error();

 private:
  /* Internally used interfaces */
  taichi::lang::Device &get() override;

  TiAotModule load_aot_module(const char *module_path) override;
  TiMemory allocate_memory(
      const taichi::lang::Device::AllocParams &params) override;
  void free_memory(TiMemory devmem) override;

  void buffer_copy(const taichi::lang::DevicePtr &dst,
                   const taichi::lang::DevicePtr &src,
                   size_t size) override;

  void submit() override;

  void wait() override;

 private:
  taichi::uint64 *result_buffer{nullptr};
  std::unique_ptr<taichi::lang::LlvmRuntimeExecutor> executor_{nullptr};
  std::unique_ptr<taichi::lang::MemoryPool> memory_pool_{nullptr};
  std::unique_ptr<taichi::lang::CompileConfig> cfg_{nullptr};
};

}  // namespace capi
