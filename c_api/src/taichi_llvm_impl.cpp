#include "taichi_core_impl.h"
#include "taichi_llvm_impl.h"

#ifdef TI_WITH_LLVM

#include "taichi/runtime/llvm/llvm_runtime_executor.h"

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

taichi::lang::DeviceAllocation LlvmRuntime::allocate_memory(
    const taichi::lang::Device::AllocParams &params) {
  taichi::lang::CompileConfig *config = executor_->get_config();
  taichi::lang::TaichiLLVMContext *tlctx =
      executor_->get_llvm_context(config->arch);
  taichi::lang::LLVMRuntime *llvm_runtime = executor_->get_llvm_runtime();
  taichi::lang::LlvmDevice *llvm_device = executor_->llvm_device();

  return llvm_device->allocate_memory_runtime(
      {params, config->ndarray_use_cached_allocator, tlctx->runtime_jit_module,
       llvm_runtime, result_buffer});
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
