#include "llvm_program.h"

#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/program/arch.h"

namespace taichi {
namespace lang {
LlvmProgramImpl::LlvmProgramImpl(CompileConfig config) : config(config) {
  snode_tree_buffer_manager = std::make_unique<SNodeTreeBufferManager>(this);

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
