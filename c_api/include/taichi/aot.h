#include "taichi/common/platform_macros.h"

#include "c_api/include/taichi/aot_vulkan.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct AotModule;
typedef struct TaichiKernel;
typedef struct TaichiRuntimeContext;
typedef struct DeviceAllocation;

TI_DLL_EXPORT AotModule *make_vulkan_aot_module(const char *module_path,
                                                VulkanRuntime runtime);
TI_DLL_EXPORT void destroy_vulkan_aot_module(AotModule *m);

TI_DLL_EXPORT TaichiKernel *get_taichi_kernel(AotModule *m, const char *name);
TI_DLL_EXPORT void launch_taichi_kernel(TaichiKernel *k,
                                        TaichiRuntimeContext *ctx);

TI_DLL_EXPORT TaichiRuntimeContext *make_runtime_context();
TI_DLL_EXPORT void destroy_runtime_context(TaichiRuntimeContext *ctx);

TI_DLL_EXPORT void set_runtime_context_arg_i32(TaichiRuntimeContext *ctx,
                                               int i,
                                               int32_t val);
TI_DLL_EXPORT void set_runtime_context_arg_float(TaichiRuntimeContext *ctx,
                                                 int i,
                                                 float val);
TI_DLL_EXPORT void set_runtime_context_arg_devalloc(
    TaichiRuntimeContext *ctx,
    int i,
    DeviceAllocation *dev_alloc);

#ifdef __cplusplus
}  // extern "C"
#endif
