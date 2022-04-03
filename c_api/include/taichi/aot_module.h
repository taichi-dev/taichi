#pragma once

#include <stdint.h>

#include "c_api/include/taichi/runtime.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AotModule;

TI_DLL_EXPORT TaichiKernel *get_taichi_kernel(AotModule *m, const char *name);

#ifdef __cplusplus
}  // extern "C"
#endif
