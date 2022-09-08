#include "taichi_core_impl.h"
#include "taichi_llvm_impl.h"

#include "taichi/program/compile_config.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
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

void LlvmRuntime::check_runtime_error() {
  executor_->check_runtime_error(this->result_buffer);
}

taichi::lang::Device &LlvmRuntime::get() {
  taichi::lang::Device *device = executor_->get_compute_device();
  return *device;
}

TiMemory LlvmRuntime::allocate_memory(
    const taichi::lang::Device::AllocParams &params) {
  taichi::lang::CompileConfig *config = executor_->get_config();
  taichi::lang::TaichiLLVMContext *tlctx =
      executor_->get_llvm_context(config->arch);
  taichi::lang::LLVMRuntime *llvm_runtime = executor_->get_llvm_runtime();
  taichi::lang::LlvmDevice *llvm_device = executor_->llvm_device();

  taichi::lang::DeviceAllocation devalloc =
      llvm_device->allocate_memory_runtime(
          {params, config->ndarray_use_cached_allocator,
           tlctx->runtime_jit_module, llvm_runtime, result_buffer});
  return devalloc2devmem(*this, devalloc);
}

void LlvmRuntime::free_memory(TiMemory devmem) {
  taichi::lang::CompileConfig *config = executor_->get_config();
  // For memory allocated through Device::allocate_memory_runtime(),
  // the corresponding Device::free_memory() interface has not been
  // implemented yet...
  if (taichi::arch_is_cpu(config->arch)) {
    TI_CAPI_NOT_SUPPORTED_IF(taichi::arch_is_cpu(config->arch));
  }

  Runtime::free_memory(devmem);
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
    TI_ASSERT(config->arch == taichi::Arch::cuda);
    taichi::lang::cuda::AotModuleParams aot_params;
    aot_params.executor_ = executor_.get();
    aot_params.module_path = module_path;
    aot_module = taichi::lang::cuda::make_aot_module(aot_params);
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  /* TODO(zhanlue): expose allocate/deallocate_snode_tree_type() to C-API
     Let's initialize SNodeTrees automatically for now since SNodeTreeType isn't
     ready yet.
  */
  auto *llvm_aot_module =
      dynamic_cast<taichi::lang::LlvmAotModule *>(aot_module.get());
  TI_ASSERT(llvm_aot_module != nullptr);
  for (size_t i = 0; i < llvm_aot_module->get_num_snode_trees(); i++) {
    auto *snode_tree = aot_module->get_snode_tree(std::to_string(i));
    taichi::lang::allocate_aot_snode_tree_type(aot_module.get(), snode_tree,
                                               this->result_buffer);
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
  // (penguinliong) Submit in LLVM backends is a nop atm.
  // TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::wait() {
  executor_->synchronize();
}

}  // namespace capi
