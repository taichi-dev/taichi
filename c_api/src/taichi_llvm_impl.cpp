#include "taichi_core_impl.h"
#include "taichi_llvm_impl.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"

#ifdef TI_WITH_LLVM

namespace capi {

LlvmRuntime::LlvmRuntime(taichi::Arch arch) : Runtime(arch) {
  taichi::lang::CompileConfig cfg;
  cfg.arch = arch;
  executor_ = std::make_unique<taichi::lang::LlvmRuntimeExecutor>(cfg, nullptr);

  taichi::lang::Device *compute_device = executor_->get_compute_device();
  memory_pool_ =
      taichi::arch_is_cpu(arch)
          ? std::make_unique<taichi::lang::MemoryPool>(arch, compute_device)
          : nullptr;

  // materialize_runtime() takes in a uint64_t** (pointer object's address) and
  // modifies the address it points to.
  //
  // Therefore we can't use host_result_buffer_.data() here,
  // since it returns a temporary copy of the internal data pointer,
  // thus we won't be able to modify the address where the std::array's data
  // pointer is pointing to.
  executor_->materialize_runtime(memory_pool_.get(), nullptr /*kNoProfiler*/,
                                 &result_buffer);
}

taichi::lang::Device &LlvmRuntime::get() {
  taichi::lang::Device *device = executor_->get_compute_device();
  return *device;
}

TiAotModule LlvmRuntime::load_aot_module(const char *module_path) {
  TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::buffer_copy(const taichi::lang::DevicePtr &dst,
                              const taichi::lang::DevicePtr &src,
                              size_t size) {
  TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::submit() {
  TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::wait() {
  TI_NOT_IMPLEMENTED;
}

}  // namespace capi

#endif  // TI_WITH_LLVM
