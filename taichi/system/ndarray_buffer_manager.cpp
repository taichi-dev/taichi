#include "ndarray_buffer_manager.h"
#include "taichi/program/program.h"
#include "taichi/llvm/llvm_program.h"

TLANG_NAMESPACE_BEGIN

NdarrayBufferManager::NdarrayBufferManager(LlvmProgramImpl *prog)
    : prog_(prog) {
  TI_TRACE("Ndarray buffer manager created.");
}

Ptr NdarrayBufferManager::allocate(JITModule *runtime_jit,
                                   void *runtime,
                                   std::size_t size,
                                   std::size_t alignment,
                                   uint64 *result_buffer) {
  TI_TRACE("allocating memory for Ndarray");
  runtime_jit->call<void *, std::size_t, std::size_t>(
      "runtime_memory_allocate_aligned", runtime, size, alignment);
  auto ptr = prog_->fetch_result<Ptr>(taichi_result_buffer_runtime_query_id,
                                      result_buffer);
  return ptr;
}

TLANG_NAMESPACE_END
