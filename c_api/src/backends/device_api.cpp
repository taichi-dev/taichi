#include "c_api/include/taichi/backends/device_api.h"

#include "taichi/backends/device.h"
#include "taichi/aot/module_loader.h"

namespace {

#include "c_api/src/inc/runtime_casts.inc.h"

}  // namespace

Taichi_DeviceAllocation *taichi_allocate_device_memory(
    Taichi_Device *dev,
    const Taichi_DeviceAllocParams *params) {
  tl::Device::AllocParams aparams;
  aparams.size = params->size;
  aparams.host_write = params->host_write;
  aparams.host_read = params->host_read;
  aparams.export_sharing = params->export_sharing;
  aparams.usage = tl::AllocUsage::Storage;
  auto *res = new tl::DeviceAllocation();
  *res = cppcast(dev)->allocate_memory(aparams);
  return reinterpret_cast<Taichi_DeviceAllocation *>(res);
}

void taichi_deallocate_device_memory(Taichi_Device *dev,
                                     Taichi_DeviceAllocation *da) {
  auto *alloc = cppcast(da);
  cppcast(dev)->dealloc_memory(*alloc);
  delete alloc;
}

void *taichi_map_device_allocation(Taichi_Device *dev,
                                   Taichi_DeviceAllocation *da) {
  tl::DeviceAllocation *alloc = cppcast(da);
  return cppcast(dev)->map(*alloc);
}

void taichi_unmap_device_allocation(Taichi_Device *dev,
                                    Taichi_DeviceAllocation *da) {
  tl::DeviceAllocation *alloc = cppcast(da);
  cppcast(dev)->unmap(*alloc);
}
