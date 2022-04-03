#include "c_api/include/taichi/runtime.h"

#include "c_api/include/taichi/aot_module.h"
#include "taichi/aot/module_loader.h"

#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"

namespace {

#include "c_api/src/inc/runtime_casts.inc.h"

tl::RuntimeContext *cppcast(TaichiRuntimeContext *ctx) {
  return reinterpret_cast<tl::RuntimeContext *>(ctx);
}

}  // namespace

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

void set_runtime_context_arg_i32(TaichiRuntimeContext *ctx,
                                 int i,
                                 int32_t val) {
  cppcast(ctx)->set_arg(i, val);
}

void set_runtime_context_arg_float(TaichiRuntimeContext *ctx,
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
