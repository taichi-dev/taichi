#include "llvm_program.h"

#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/program/arch.h"
#include "taichi/platform/cuda/detect_cuda.h"

namespace taichi {
namespace lang {
LlvmProgramImpl::LlvmProgramImpl(CompileConfig &config_) {
  runtime_mem_info = Runtime::create(config_.arch);
  if (config_.arch == Arch::cuda) {
    if (!runtime_mem_info) {
      TI_WARN("Taichi is not compiled with CUDA.");
      config_.arch = host_arch();
    } else if (!is_cuda_api_available()) {
      TI_WARN("No CUDA driver API detected.");
      config_.arch = host_arch();
    } else if (!runtime_mem_info->detected()) {
      TI_WARN("No CUDA device detected.");
      config_.arch = host_arch();
    } else {
      // CUDA runtime created successfully
    }
    if (config_.arch != Arch::cuda) {
      TI_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  }
  config = config_;
  snode_tree_buffer_manager = std::make_unique<SNodeTreeBufferManager>(this);

  thread_pool = std::make_unique<ThreadPool>(config.cpu_max_num_threads);
  llvm_context_host = std::make_unique<TaichiLLVMContext>(host_arch());
}

void LlvmProgramImpl::device_synchronize() {
  if (config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    TI_ERROR("No CUDA support");
#endif
  }
}

uint64 LlvmProgramImpl::fetch_result_uint64(int i, uint64 *result_buffer) {
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  device_synchronize();
  uint64 ret;
  auto arch = config.arch;
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    if (config.use_unified_memory) {
      // More efficient than a cudaMemcpy call in practice
      ret = result_buffer[i];
    } else {
      CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i,
                                                       sizeof(uint64));
    }
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else {
    ret = result_buffer[i];
  }
  return ret;
}

}  // namespace lang
}  // namespace taichi
