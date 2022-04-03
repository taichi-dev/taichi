#pragma once

#include <stdint.h>

#include "c_api/include/taichi/runtime.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AotModule AotModule;

TI_DLL_EXPORT TaichiKernel *get_taichi_kernel(AotModule *m, const char *name);

TI_DLL_EXPORT size_t get_root_size_from_aot_module(AotModule *m);

#ifdef __cplusplus
}  // extern "C"
#endif
