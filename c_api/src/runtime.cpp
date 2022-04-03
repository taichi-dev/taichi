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
                                 int param_i,
                                 int32_t val) {
  cppcast(ctx)->set_arg(param_i, val);
}

void set_runtime_context_arg_float(TaichiRuntimeContext *ctx,
                                   int param_i,
                                   float val) {
  cppcast(ctx)->set_arg(param_i, val);
}

void set_runtime_context_arg_ndarray(TaichiRuntimeContext *ctx,
                                     int param_i,
                                     DeviceAllocation *arr,
                                     const NdShape *shape,
                                     const NdShape *elem_shape) {
  tl::RuntimeContext *rctx = cppcast(ctx);
  rctx->set_arg(param_i, cppcast(arr));
  rctx->set_device_allocation(param_i, /*is_device_allocation=*/true);
  int extra_arg_i = 0;
  for (int i = 0; i < shape->length; ++i) {
    rctx->extra_args[param_i][extra_arg_i++] = shape->data[i];
  }
  if (elem_shape) {
    for (int i = 0; i < elem_shape->length; ++i) {
      rctx->extra_args[param_i][extra_arg_i++] = elem_shape->data[i];
    }
  }
}

void set_runtime_context_arg_scalar_ndarray(TaichiRuntimeContext *ctx,
                                            int param_i,
                                            DeviceAllocation *arr,
                                            const NdShape *shape) {
  set_runtime_context_arg_ndarray(ctx, param_i, arr, shape,
                                  /*elem_shape=*/NULL);
}

#undef TI_RUNTIME_HOST
