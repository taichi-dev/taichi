#include "taichi/taichi.h"
#include "taichi/aot/module_loader.h"
#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/runtime/vulkan/runtime.h"

#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"

void tiLaunchKernel(TiKernel k, TiContext ctx) {
  auto *kn = reinterpret_cast<taichi::lang::aot::Kernel *>(k);
  kn->launch((taichi::lang::RuntimeContext*)(ctx));
}

TiContext tiCreateContext() {
  auto *ctx = new taichi::lang::RuntimeContext();
  memset(ctx, 0, sizeof(taichi::lang::RuntimeContext));
  return reinterpret_cast<TiContext *>(ctx);
}

void tiDestroyContext(TiContext ctx) {
  delete (taichi::lang::RuntimeContext*)(ctx);
}



void tiSetContextArgumentI32(TiContext ctx,
                             int param_i,
                             int32_t val) {
  ((taichi::lang::RuntimeContext*)ctx)->set_arg(param_i, val);
}

void tiSetContextArgumentF32(TiContext ctx,
                             int param_i,
                             float val) {
  ((taichi::lang::RuntimeContext*)ctx)->set_arg(param_i, val);
}

void tiSetContextArgumentNdArray(TiContext *ctx,
                                 int param_i,
                                 TiDeviceAllocation *arr,
                                 const TiNdShape *shape,
                                 const TiNdShape *elem_shape) {
  taichi::lang::RuntimeContext *rctx = ((taichi::lang::RuntimeContext*)ctx);
  std::vector<int> shape2(
    shape->data, shape->data + shape->length);
  std::vector<int> elem_shape2(
    elem_shape->data, elem_shape->data + elem_shape->length);
  rctx->set_arg_devalloc(
    param_i, *(taichi::lang::DeviceAllocation*)(arr), shape2, elem_shape2);
}



TiKernel tiGetAotModuleKernel(TiAotModule m, const char *name) {
  auto *mod = (taichi::lang::aot::Module*)m;
  auto *k = mod->get_kernel(name);
  return (void*)k;
}

size_t tiGetAotModuleRootBufferSize(TiAotModule m) {
  return ((taichi::lang::aot::Module*)m)->get_root_size();
}



TiDeviceAllocation tiCreateDeviceAllocation(
    TiDevice dev,
    const TiDeviceAlloctionInfo *params) {
  taichi::lang::Device::AllocParams aparams;
  aparams.size = params->size;
  aparams.host_write = params->host_write;
  aparams.host_read = params->host_read;
  aparams.export_sharing = params->export_sharing;
  aparams.usage = taichi::lang::AllocUsage::Storage;
  auto *res = new taichi::lang::DeviceAllocation();
  *res = ((taichi::lang::Device*)dev)->allocate_memory(aparams);
  return reinterpret_cast<TiDeviceAllocation *>(res);
}

void tiDestroyDeviceAllocation(TiDevice dev,
                                     TiDeviceAllocation da) {
  auto *alloc = (taichi::lang::DeviceAllocation*)da;
  ((taichi::lang::Device*)dev)->dealloc_memory(*alloc);
  delete alloc;
}

void *tiMapDeviceAllocation(TiDevice dev,
                                   TiDeviceAllocation da) {
  auto *alloc = (taichi::lang::DeviceAllocation*)da;
  return ((taichi::lang::Device*)dev)->map(*alloc);
}

void tiUnmapDeviceAllocation(TiDevice dev,
                                    TiDeviceAllocation da) {
  auto *alloc = (taichi::lang::DeviceAllocation*)da;
  ((taichi::lang::Device*)dev)->unmap(*alloc);
}

#undef TI_RUNTIME_HOST
