#include "taichi/rhi/llvm/llvm_device.h"
#include "taichi/jit/jit_module.h"

namespace taichi::lang {

uint64_t *LlvmDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size, params.result_buffer);
  return reinterpret_cast<uint64_t *>(params.result_buffer[0]);
}

}  // namespace taichi::lang
