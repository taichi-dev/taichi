#ifdef TI_WITH_LLVM

#include "taichi_core_impl.h"
#include "taichi_llvm_impl.h"

#include "taichi/taichi_cpu.h"
#include "taichi/taichi_cuda.h"

#include "taichi/program/compile_config.h"
#include "taichi/runtime/llvm/llvm_runtime_executor.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"
#include "taichi/runtime/cpu/kernel_launcher.h"
#include "taichi/rhi/cpu/cpu_device.h"

#ifdef TI_WITH_CUDA
#include "taichi/rhi/cuda/cuda_device.h"
#include "taichi/runtime/cuda/kernel_launcher.h"
#endif

namespace capi {

LlvmRuntime::LlvmRuntime(taichi::Arch arch) : Runtime(arch) {
  cfg_ = std::make_unique<taichi::lang::CompileConfig>();
  cfg_->arch = arch;

  executor_ =
      std::make_unique<taichi::lang::LlvmRuntimeExecutor>(*cfg_.get(), nullptr);

  // materialize_runtime() takes in a uint64_t** (pointer object's address) and
  // modifies the address it points to.
  //
  // Therefore we can't use host_result_buffer_.data() here,
  // since it returns a temporary copy of the internal data pointer,
  // thus we won't be able to modify the address where the std::array's data
  // pointer is pointing to.
  executor_->materialize_runtime(nullptr /*kNoProfiler*/, &result_buffer);
}

LlvmRuntime::~LlvmRuntime() {
  executor_.reset();
  cfg_.reset();
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
  taichi::lang::LLVMRuntime *llvm_runtime = executor_->get_llvm_runtime();
  taichi::lang::LlvmDevice *llvm_device = executor_->llvm_device();

  taichi::lang::DeviceAllocation devalloc =
      llvm_device->allocate_memory_runtime({params,
                                            executor_->get_runtime_jit_module(),
                                            llvm_runtime, result_buffer});
  return devalloc2devmem(*this, devalloc);
}

void LlvmRuntime::free_memory(TiMemory devmem) {
  const taichi::lang::CompileConfig &config = executor_->get_config();
  // For memory allocated through Device::allocate_memory_runtime(),
  // the corresponding Device::free_memory() interface has not been
  // implemented yet...
  if (taichi::arch_is_cpu(config.arch)) {
    TI_CAPI_NOT_SUPPORTED_IF(taichi::arch_is_cpu(config.arch));
  }

  Runtime::free_memory(devmem);
}

TiAotModule LlvmRuntime::load_aot_module(const char *module_path) {
  const auto &config = executor_->get_config();
  std::unique_ptr<taichi::lang::aot::Module> aot_module{nullptr};

  if (taichi::arch_is_cpu(config.arch)) {
    taichi::lang::cpu::KernelLauncher::Config cfg;
    cfg.executor = executor_.get();
    taichi::lang::LLVM::AotModuleParams aot_params;
    aot_params.executor_ = executor_.get();
    aot_params.kernel_launcher =
        std::make_unique<taichi::lang::cpu::KernelLauncher>(std::move(cfg));
    aot_params.module_path = module_path;
    aot_module = taichi::lang::LLVM::make_aot_module(std::move(aot_params));
  } else {
#ifdef TI_WITH_CUDA
    TI_ASSERT(config.arch == taichi::Arch::cuda);
    taichi::lang::cuda::KernelLauncher::Config cfg;
    cfg.executor = executor_.get();
    taichi::lang::LLVM::AotModuleParams aot_params;
    aot_params.executor_ = executor_.get();
    aot_params.kernel_launcher =
        std::make_unique<taichi::lang::cuda::KernelLauncher>(std::move(cfg));
    aot_params.module_path = module_path;
    aot_module = taichi::lang::LLVM::make_aot_module(std::move(aot_params));
#else
    TI_NOT_IMPLEMENTED;
#endif
  }

  /* TODO(zhanlue): expose allocate/deallocate_snode_tree_type() to C-API
     Let's initialize SNodeTrees automatically for now since SNodeTreeType isn't
     ready yet.
  */
  auto *llvm_aot_module =
      dynamic_cast<taichi::lang::LLVM::LlvmAotModule *>(aot_module.get());
  TI_ASSERT(llvm_aot_module != nullptr);
  for (size_t i = 0; i < llvm_aot_module->get_num_snode_trees(); i++) {
    auto *snode_tree = aot_module->get_snode_tree(std::to_string(i));
    taichi::lang::LLVM::allocate_aot_snode_tree_type(
        aot_module.get(), snode_tree, this->result_buffer);
  }

  return (TiAotModule)(new AotModule(*this, std::move(aot_module)));
}

void LlvmRuntime::buffer_copy(const taichi::lang::DevicePtr &dst,
                              const taichi::lang::DevicePtr &src,
                              size_t size) {
  TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::flush() {
  // (penguinliong) Flush in LLVM backends is a nop atm.
  // TI_NOT_IMPLEMENTED;
}

void LlvmRuntime::wait() {
  executor_->synchronize();
}

}  // namespace capi

// function.export_cpu_runtime
void ti_export_cpu_memory(TiRuntime runtime,
                          TiMemory memory,
                          TiCpuMemoryInteropInfo *interop_info) {
#ifdef TI_WITH_LLVM
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(memory);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  if (((Runtime *)runtime)->arch != taichi::Arch::x64 &&
      ((Runtime *)runtime)->arch != taichi::Arch::arm64) {
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!= cpu");
    return;
  }

  capi::LlvmRuntime *llvm_runtime =
      static_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  taichi::lang::DeviceAllocation devalloc =
      devmem2devalloc(*llvm_runtime, memory);

  auto &device = llvm_runtime->get();
  auto &cpu_device = static_cast<taichi::lang::cpu::CpuDevice &>(device);

  auto cpu_info = cpu_device.get_alloc_info(devalloc);

  interop_info->ptr = cpu_info.ptr;
  interop_info->size = cpu_info.size;
#else
  TI_NOT_IMPLEMENTED;
#endif  // TI_WITH_LLVM
}

// function.export_cuda_runtime
void ti_export_cuda_memory(TiRuntime runtime,
                           TiMemory memory,
                           TiCudaMemoryInteropInfo *interop_info) {
#ifdef TI_WITH_CUDA
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(memory);
  TI_CAPI_ARGUMENT_NULL(interop_info);
  if (((Runtime *)runtime)->arch != taichi::Arch::cuda) {
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!= cuda");
    return;
  }

  auto *llvm_runtime = static_cast<capi::LlvmRuntime *>((Runtime *)runtime);
  taichi::lang::DeviceAllocation devalloc =
      devmem2devalloc(*llvm_runtime, memory);

  auto &device = llvm_runtime->get();
  auto &cuda_device = static_cast<taichi::lang::cuda::CudaDevice &>(device);

  auto cuda_info = cuda_device.get_alloc_info(devalloc);

  interop_info->ptr = cuda_info.ptr;
  interop_info->size = cuda_info.size;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

#endif  // TI_WITH_LLVM
