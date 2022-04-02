#include "c_api/include/taichi/aot.h"

#include "taichi/backends/vulkan/aot_module_loader_impl.h"

#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"

namespace {
namespace tl = taichi::lang;

tl::aot::Module *cppcast(AotModule *m) {
  return reinterpret_cast<tl::aot::Module *>(m);
}

tl::DeviceAllocation *cppcast(DeviceAllocation *da) {
  return reinterpret_cast<tl::DeviceAllocation *>(da);
}

tl::RuntimeContext *cppcast(TaichiRuntimeContext *ctx) {
  return reinterpret_cast<tl::RuntimeContext *>(ctx);
}

}  // namespace

AotModule *make_vulkan_aot_module(const char *module_path,
                                  VulkanRuntime *runtime) {
  tl::vulkan::AotModuleParams params;
  params.module_path = module_path;
  params.runtime = reinterpret_cast<tl::vulkan::VkRuntime *>(runtime);
  auto mod = tl::vulkan::make_aot_module(params);
  return reinterpret_cast<AotModule *>(mod.release());
}

void destroy_vulkan_aot_module(AotModule *m) {
  delete cppcast(m);
}

TaichiKernel *get_taichi_kernel(AotModule *m, const char *name) {
  auto *mod = cppcast(m);
  auto *k = mod->get_kernel(name);
  return reinterpret_cast<TaichiKernel *>(k);
}

void launch_taichi_kernel(TaichiKernel *k, TaichiRuntimeContext *ctx) {
  auto *kn = reinterpret_cast<tl::aot::Kernel *>(k);
  kn->launch(cppcast(ctx));
}

TaichiRuntimeContext *make_runtime_context() {
  auto *ctx = new tl::RuntimeContext();
  memset(ctx, 0, sizeof(tl::RuntimeContext));
  return reinterpret_cast<TaichiRuntimeContext *>(ctx);
}

void destroy_runtime_context(TaichiRuntimeContext *ctx) {
  delete cppcast(ctx);
}

TI_DLL_EXPORT void set_runtime_context_arg_i32(TaichiRuntimeContext *ctx,
                                               int i,
                                               int32_t val) {
  cppcast(ctx)->set_arg(i, val);
}

TI_DLL_EXPORT void set_runtime_context_arg_float(TaichiRuntimeContext *ctx,
                                                 int i,
                                                 float val) {
  cppcast(ctx)->set_arg(i, val);
}

TI_DLL_EXPORT void set_runtime_context_arg_devalloc(
    TaichiRuntimeContext *ctx,
    int i,
    DeviceAllocation *dev_alloc) {
  cppcast(ctx)->set_arg(i, cppcast(dev_alloc));
  cppcast(ctx)->set_device_allocation(i, /*is_device_allocation=*/true);
}

#undef TI_RUNTIME_HOST
