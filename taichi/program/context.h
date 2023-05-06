#pragma once

// Use relative path here for runtime compilation
#include "taichi/inc/constants.h"
#include <cstdint>

#if defined(TI_RUNTIME_HOST)
namespace taichi::lang {
#endif

struct LLVMRuntime;
// "RuntimeContext" holds necessary data for kernel body execution, such as a
// pointer to the LLVMRuntime struct, kernel arguments, and the thread id (if on
// CPU).
struct RuntimeContext {
  char *arg_buffer{nullptr};

  LLVMRuntime *runtime{nullptr};

  int32_t cpu_thread_id;

  // We move the pointer of result buffer from LLVMRuntime to RuntimeContext
  // because each real function need a place to store its result, but
  // LLVMRuntime is shared among functions. So we moved the pointer to
  // RuntimeContext which each function have one.
  uint64_t *result_buffer;
};

#if defined(TI_RUNTIME_HOST)
}  // namespace taichi::lang
#endif
