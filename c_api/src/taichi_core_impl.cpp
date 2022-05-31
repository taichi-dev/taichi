#include "taichi_core_impl.h"
#include "taichi_vulkan_impl.h"

taichi::lang::DeviceAllocation devmem2devalloc(Device &device,
                                               TiDeviceMemory devmem) {
  return taichi::lang::DeviceAllocation{
      &device.get(), (taichi::lang::DeviceAllocationId)((size_t)devmem)};
}

Device::Device(taichi::Arch arch) : arch(arch) {
}
Device::~Device() {
  TI_NOT_IMPLEMENTED;
}

VulkanDevice *Device::as_vk() {
  TI_ASSERT(arch == taichi::Arch::vulkan);
  return static_cast<VulkanDevice *>(this);
}

Context::Context(Device &device) : device_(&device), runtime_context_{} {
}
Context::~Context() {
  TI_NOT_IMPLEMENTED;
}
taichi::lang::RuntimeContext &Context::get() {
  return runtime_context_;
}
Device &Context::device() {
  return *device_;
}

VulkanContext *Context::as_vk() {
  TI_ASSERT(device().arch == taichi::Arch::vulkan);
  return static_cast<VulkanContext *>(this);
}

AotModule::AotModule(Context &context,
                     std::unique_ptr<taichi::lang::aot::Module> &&aot_module)
    : context_(&context),
      aot_module_(std::forward<std::unique_ptr<taichi::lang::aot::Module>>(
          aot_module)) {
}
taichi::lang::aot::Module& AotModule::get() {
  return *aot_module_;
}
Context &AotModule::context() {
  return *context_;
}

// -----------------------------------------------------------------------------

TiDevice tiCreateDevice(TiArch arch) {
  switch (arch) {
#ifdef TI_WITH_VULKAN
    case TI_ARCH_VULKAN:
      return static_cast<Device *>(new VulkanDeviceOwned);
#endif  // TI_WITH_VULKAN
    default:
      TI_ASSERT(false);
  }
  return nullptr;
}
void tiDestroyDevice(TiDevice device) {
  delete (Device *)device;
}
void tiDeviceWaitIdle(TiDevice device) {
  ((Device*)device)->get().wait_idle();
}

TiContext tiCreateContext(TiDevice device) {
  return ((Device *)device)->create_context();
}
void tiDestroyContext(TiContext context) {
  delete (Context *)context;
}

TiDeviceMemory tiAllocateMemory(TiDevice device,
                                const TiMemoryAllocateInfo *createInfo) {
  taichi::lang::AllocUsage usage{};
  if (createInfo->usage & TI_MEMORY_USAGE_STORAGE_BIT) {
    usage = usage | taichi::lang::AllocUsage::Storage;
  }
  if (createInfo->usage & TI_MEMORY_USAGE_UNIFORM_BIT) {
    usage = usage | taichi::lang::AllocUsage::Uniform;
  }
  if (createInfo->usage & TI_MEMORY_USAGE_VERTEX_BIT) {
    usage = usage | taichi::lang::AllocUsage::Vertex;
  }
  if (createInfo->usage & TI_MEMORY_USAGE_INDEX_BIT) {
    usage = usage | taichi::lang::AllocUsage::Index;
  }

  taichi::lang::Device::AllocParams params{};
  params.size = createInfo->size;
  params.host_write = createInfo->hostWrite;
  params.host_read = createInfo->hostRead;
  params.export_sharing = createInfo->exportSharing;
  params.usage = usage;
  return (TiDeviceMemory)((Device *)device)
      ->get()
      .allocate_memory(params)
      .alloc_id;
}
void tiFreeMemory(TiDevice device, TiDeviceMemory devmem) {
  Device *device2 = (Device *)device;
  device2->get().dealloc_memory(devmem2devalloc(*device2, devmem));
}

void *tiMapMemory(TiDevice device, TiDeviceMemory devmem) {
  Device *device2 = (Device *)device;
  return device2->get().map(devmem2devalloc(*device2, devmem));
}
void tiUnmapMemory(TiDevice device, TiDeviceMemory devmem) {
  Device *device2 = (Device *)device;
  device2->get().unmap(devmem2devalloc(*device2, devmem));
}

void tiDestroyAotModule(TiAotModule mod) {
  delete (AotModule *)mod;
}
TiKernel tiGetAotModuleKernel(TiAotModule mod, const char *name) {
  return (TiKernel)((AotModule *)mod)->get().get_kernel(name);
}

void tiSetContextArgNdArray(TiContext context,
                            uint32_t argIndex,
                            const TiNdArray *ndarray) {
  Context *context2 = (Context *)context;
  Device &device = context2->device();

  taichi::lang::DeviceAllocation devalloc =
      devmem2devalloc(device, ndarray->devmem);

  std::vector<int> shape(ndarray->shape.dims,
                         ndarray->shape.dims + ndarray->shape.dimCount);

  if (ndarray->elem_shape.dimCount != 0) {
    std::vector<int> elem_shape(
        ndarray->elem_shape.dims,
        ndarray->elem_shape.dims + ndarray->elem_shape.dimCount);

    context2->get().set_arg_devalloc(argIndex, devalloc, shape, elem_shape);
  } else {
    context2->get().set_arg_devalloc(argIndex, devalloc, shape);
  }
}
void tiSetContextArgI32(TiContext context, uint32_t argIndex, int32_t value) {
  ((Context *)context)->get().set_arg(argIndex, value);
}
void tiSetContextArgF32(TiContext context, uint32_t argIndex, float value) {
  ((Context *)context)->get().set_arg(argIndex, value);
}
void tiLaunchKernel(TiContext context, TiKernel kernel) {
  ((taichi::lang::aot::Kernel *)kernel)->launch(&((Context *)context)->get());
}
