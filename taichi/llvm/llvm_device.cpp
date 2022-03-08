#include "taichi/llvm/llvm_device.h"

namespace taichi {
namespace lang {

uint64_t *LlvmDevice::allocate_llvm_runtime_memory_jit(
    const LlvmRuntimeAllocParams &params) {
  params.runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", params.runtime, params.size,
      taichi_page_size);
  return taichi_union_cast_with_different_sizes<uint64_t *>(fetch_result_uint64(
      taichi_result_buffer_runtime_query_id, params.result_buffer));
}

}  // namespace lang
}  // namespace taichi
