#include "taichi_core_impl.h"
#include "taichi_llvm_impl.h"

#ifdef TI_WITH_LLVM

#include "taichi/program/compile_config.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/cpu/aot_module_loader_impl.h"

#ifdef TI_WITH_CUDA
#include "taichi/runtime/cuda/aot_module_loader_impl.h"
#endif

namespace capi {

LlvmRuntime::LlvmRuntime(taichi::Arch arch) : Runtime(arch) {
  cfg_ = std::make_unique<taichi::lang::CompileConfig>();
  cfg_->arch = arch;

  executor_ =
      std::make_unique<taichi::lang::LlvmRuntimeExecutor>(*cfg_.get(), nullptr);

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

void LlvmRuntime::deallocate_memory(TiMemory devmem) {
  taichi::lang::CompileConfig *config = executor_->get_config();
  if (taichi::arch_is_cpu(config->arch)) {
    // For memory allocated through Device::allocate_memory_runtime(),
    // the corresponding Device::deallocate_memory() interface has not been
    // implemented yet...
    TI_NOT_IMPLEMENTED;
  }

  Runtime::deallocate_memory(devmem);
}

TiAotModule LlvmRuntime::load_aot_module(const char *module_path) {
  auto *config = executor_->get_config();
  std::unique_ptr<taichi::lang::aot::Module> aot_module{nullptr};

  if (taichi::arch_is_cpu(config->arch)) {
    taichi::lang::cpu::AotModuleParams aot_params;
    aot_params.executor_ = executor_.get();
    aot_params.module_path = module_path;
    aot_module = taichi::lang::cpu::make_aot_module(aot_params);

  } else {
#ifdef TI_WITH_CUDA
    std::cout << taichi::arch_name(config->arch) << std::endl;
    TI_ASSERT(config->arch == taichi::Arch::cuda);
    taichi::lang::cuda::AotModuleParams aot_params;
    aot_params.executor_ = executor_.get();
    aot_params.module_path = module_path;
    aot_module = taichi::lang::cuda::make_aot_module(aot_params);
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  // Insert LLVMRuntime to RuntimeContext
  executor_->prepare_runtime_context(&this->runtime_context_);
  return (TiAotModule)(new AotModule(*this, std::move(aot_module)));
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
  executor_->synchronize();
}

}  // namespace capi

#endif  // TI_WITH_LLVM
