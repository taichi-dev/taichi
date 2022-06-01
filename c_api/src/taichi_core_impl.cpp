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
}

VulkanDevice *Device::as_vk() {
  TI_ASSERT(arch == taichi::Arch::vulkan);
  return static_cast<VulkanDevice *>(this);
}

Context::Context(Device &device) : device_(&device), runtime_context_{} {
}
Context::~Context() {
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
taichi::lang::aot::Module &AotModule::get() {
  return *aot_module_;
}
Context &AotModule::context() {
  return *context_;
}

// -----------------------------------------------------------------------------

TiDevice ti_create_device(TiArch arch) {
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
void ti_destroy_device(TiDevice device) {
  delete (Device *)device;
}
void ti_device_wait_idle(TiDevice device) {
  ((Device *)device)->get().wait_idle();
}

TiContext ti_create_context(TiDevice device) {
  return ((Device *)device)->create_context();
}
void ti_destroy_context(TiContext context) {
  delete (Context *)context;
}

TiDeviceMemory ti_allocate_memory(TiDevice device,
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
  params.host_write = createInfo->host_write;
  params.host_read = createInfo->host_read;
  params.export_sharing = createInfo->export_sharing;
  params.usage = usage;
  return (TiDeviceMemory)((Device *)device)
      ->get()
      .allocate_memory(params)
      .alloc_id;
}
void ti_free_memory(TiDevice device, TiDeviceMemory devmem) {
  Device *device2 = (Device *)device;
  device2->get().dealloc_memory(devmem2devalloc(*device2, devmem));
}

void *ti_map_memory(TiDevice device, TiDeviceMemory devmem) {
  Device *device2 = (Device *)device;
  return device2->get().map(devmem2devalloc(*device2, devmem));
}
void ti_unmap_memory(TiDevice device, TiDeviceMemory devmem) {
  Device *device2 = (Device *)device;
  device2->get().unmap(devmem2devalloc(*device2, devmem));
}

void ti_destroy_aot_module(TiAotModule mod) {
  delete (AotModule *)mod;
}
TiKernel ti_get_aot_module_kernel(TiAotModule mod, const char *name) {
  return (TiKernel)((AotModule *)mod)->get().get_kernel(name);
}

void ti_set_context_arg_ndarray(TiContext context,
                                uint32_t arg_index,
                                const TiNdArray *ndarray) {
  Context *context2 = (Context *)context;
  Device &device = context2->device();

  taichi::lang::DeviceAllocation devalloc =
      devmem2devalloc(device, ndarray->devmem);

  std::vector<int> shape(ndarray->shape.dims,
                         ndarray->shape.dims + ndarray->shape.dim_count);

  if (ndarray->elem_shape.dim_count != 0) {
    std::vector<int> elem_shape(
        ndarray->elem_shape.dims,
        ndarray->elem_shape.dims + ndarray->elem_shape.dim_count);

    context2->get().set_arg_devalloc(arg_index, devalloc, shape, elem_shape);
  } else {
    context2->get().set_arg_devalloc(arg_index, devalloc, shape);
  }
}
void ti_set_context_arg_i32(TiContext context,
                            uint32_t arg_index,
                            int32_t value) {
  ((Context *)context)->get().set_arg(arg_index, value);
}
void ti_set_context_arg_f32(TiContext context,
                            uint32_t arg_index,
                            float value) {
  ((Context *)context)->get().set_arg(arg_index, value);
}
void ti_launch_kernel(TiContext context, TiKernel kernel) {
  ((taichi::lang::aot::Kernel *)kernel)->launch(&((Context *)context)->get());
}
