#pragma once

#ifdef TI_WITH_LLVM
#include "taichi_core_impl.h"

namespace taichi {
namespace lang {
class LlvmRuntimeExecutor;
class MemoryPool;
}  // namespace lang
}  // namespace taichi

namespace capi {

class LlvmRuntime : public Runtime {
 public:
  LlvmRuntime(taichi::Arch arch);

 private:
  /* Internally used interfaces */
  taichi::lang::Device &get() override;

  TiAotModule load_aot_module(const char *module_path) override;

  void buffer_copy(const taichi::lang::DevicePtr &dst,
                   const taichi::lang::DevicePtr &src,
                   size_t size) override;

  void submit() override;

  void wait() override;

 private:
  taichi::uint64 *result_buffer{nullptr};
  std::unique_ptr<taichi::lang::LlvmRuntimeExecutor> executor_{nullptr};
  std::unique_ptr<taichi::lang::MemoryPool> memory_pool_{nullptr};
};

}  // namespace capi

#endif  // TI_WITH_LLVM
