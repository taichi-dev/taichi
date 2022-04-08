#pragma once

#include <stdint.h>

#include "c_api/include/taichi/runtime.h"
#include "taichi/common/platform_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Taichi_AotModule Taichi_AotModule;

TI_DLL_EXPORT Taichi_Kernel *taichi_get_kernel_from_aot_module(
    Taichi_AotModule *m,
    const char *name);

TI_DLL_EXPORT size_t taichi_get_root_size_from_aot_module(Taichi_AotModule *m);

#ifdef __cplusplus
}  // extern "C"
#endif
