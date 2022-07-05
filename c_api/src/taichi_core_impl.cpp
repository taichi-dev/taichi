#include "taichi_core_impl.h"
#include "taichi_vulkan_impl.h"
#include "taichi/program/ndarray.h"

Runtime::Runtime(taichi::Arch arch) : arch(arch) {
}
Runtime::~Runtime() {
}

VulkanRuntime *Runtime::as_vk() {
  TI_ASSERT(arch == taichi::Arch::vulkan);
#ifdef TI_WITH_VULKAN
  return static_cast<VulkanRuntime *>(this);
#else
  return nullptr;
#endif
}

AotModule::AotModule(Runtime &runtime,
                     std::unique_ptr<taichi::lang::aot::Module> &&aot_module)
    : runtime_(&runtime), aot_module_(std::move(aot_module)) {
}

taichi::lang::aot::CompiledGraph &AotModule::get_cgraph(
    const std::string &name) {
  auto it = loaded_cgraphs_.find(name);
  if (it == loaded_cgraphs_.end()) {
    return *loaded_cgraphs_
                .emplace(std::make_pair(name, aot_module_->get_graph(name)))
                .first->second;
  } else {
    return *it->second;
  }
}
taichi::lang::aot::Module &AotModule::get() {
  return *aot_module_;
}
Runtime &AotModule::runtime() {
  return *runtime_;
}

// -----------------------------------------------------------------------------

TiRuntime ti_create_runtime(TiArch arch) {
  switch (arch) {
#ifdef TI_WITH_VULKAN
    case TI_ARCH_VULKAN:
      return (TiRuntime)(static_cast<Runtime *>(new VulkanRuntimeOwned));
#endif  // TI_WITH_VULKAN
    default:
      TI_WARN("ignored attempt to create runtime on unknown arch");
      return TI_NULL_HANDLE;
  }
  return TI_NULL_HANDLE;
}
void ti_destroy_runtime(TiRuntime runtime) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to destroy runtime of null handle");
    return;
  }
  delete (Runtime *)runtime;
}

TiMemory ti_allocate_memory(TiRuntime runtime,
                            const TiMemoryAllocateInfo *createInfo) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to allocate memory on runtime of null handle");
    return TI_NULL_HANDLE;
  }

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
  taichi::lang::DeviceAllocation devalloc =
      ((Runtime *)runtime)->get().allocate_memory(params);
  return devalloc2devmem(devalloc);
}
void ti_free_memory(TiRuntime runtime, TiMemory devmem) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to free memory on runtime of null handle");
    return;
  }
  if (devmem == nullptr) {
    TI_WARN("ignored attempt to free memory of null handle");
    return;
  }

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->get().dealloc_memory(devmem2devalloc(*runtime2, devmem));
}

void *ti_map_memory(TiRuntime runtime, TiMemory devmem) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to map memory on runtime of null handle");
    return nullptr;
  }
  if (devmem == nullptr) {
    TI_WARN("ignored attempt to map memory of null handle");
    return nullptr;
  }
  Runtime *runtime2 = (Runtime *)runtime;
  return runtime2->get().map(devmem2devalloc(*runtime2, devmem));
}
void ti_unmap_memory(TiRuntime runtime, TiMemory devmem) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to unmap memory on runtime of null handle");
    return;
  }
  if (devmem == nullptr) {
    TI_WARN("ignored attempt to unmap memory of null handle");
    return;
  }
  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->get().unmap(devmem2devalloc(*runtime2, devmem));
}

void ti_copy_memory_device_to_device(TiRuntime runtime,
                                     const TiMemorySlice *dst_memory,
                                     const TiMemorySlice *src_memory) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to copy memory on runtime of null handle");
    return;
  }
  if (dst_memory == nullptr || dst_memory->memory == nullptr) {
    TI_WARN("ignored attempt to copy to dst memory of null handle");
    return;
  }
  if (src_memory == nullptr || src_memory->memory == nullptr) {
    TI_WARN("ignored attempt to copy from src memory of null handle");
    return;
  }
  if (src_memory->size != dst_memory->size) {
    TI_WARN("ignored attempt to copy memory of mismatched size");
    return;
  }
  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = devmem2devalloc(*runtime2, dst_memory->memory)
                 .get_ptr(dst_memory->offset);
  auto src = devmem2devalloc(*runtime2, src_memory->memory)
                 .get_ptr(src_memory->offset);
  runtime2->buffer_copy(dst, src, dst_memory->size);
}

TiAotModule ti_load_aot_module(TiRuntime runtime, const char *module_path) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to load aot module on runtime of null handle");
    return TI_NULL_HANDLE;
  }
  if (module_path == nullptr) {
    TI_WARN("ignored attempt to load aot module with null path");
    return TI_NULL_HANDLE;
  }

  return ((Runtime *)runtime)->load_aot_module(module_path);
}
void ti_destroy_aot_module(TiAotModule mod) {
  if (mod == nullptr) {
    TI_WARN("ignored attempt to destroy aot module of null handle");
    return;
  }

  delete (AotModule *)mod;
}
TiKernel ti_get_aot_module_kernel(TiAotModule mod, const char *name) {
  if (mod == nullptr) {
    TI_WARN("ignored attempt to get kernel from aot module of null handle");
    return TI_NULL_HANDLE;
  }
  return (TiKernel)((AotModule *)mod)->get().get_kernel(name);
}
TiComputeGraph ti_get_aot_module_compute_graph(TiAotModule mod,
                                               const char *name) {
  if (mod == nullptr) {
    TI_WARN(
        "ignored attempt to get compute graph from aot module of null handle");
    return TI_NULL_HANDLE;
  }
  AotModule *aot_module = ((AotModule *)mod);
  return (TiComputeGraph)&aot_module->get_cgraph(name);
}

void ti_launch_kernel(TiRuntime runtime,
                      TiKernel kernel,
                      uint32_t arg_count,
                      const TiArgument *args) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to launch kernel on runtime of null handle");
    return;
  }
  if (kernel == nullptr) {
    TI_WARN("ignored attempt to launch kernel of null handle");
    return;
  }

  Runtime &runtime2 = *((Runtime *)runtime);
  taichi::lang::RuntimeContext &runtime_context = runtime2.runtime_context_;
  std::vector<std::unique_ptr<taichi::lang::DeviceAllocation>> devallocs;

  for (uint32_t i = 0; i < arg_count; ++i) {
    const auto &arg = args[i];
    switch (arg.type) {
      case TI_ARGUMENT_TYPE_I32: {
        runtime_context.set_arg(i, arg.value.i32);
        break;
      }
      case TI_ARGUMENT_TYPE_F32: {
        runtime_context.set_arg(i, arg.value.f32);
        break;
      }
      case TI_ARGUMENT_TYPE_NDARRAY: {
        // Don't allocate it on stack. `DeviceAllocation` is referred to by
        // `GfxRuntime::launch_kernel`.
        std::unique_ptr<taichi::lang::DeviceAllocation> devalloc =
            std::make_unique<taichi::lang::DeviceAllocation>(
                devmem2devalloc(runtime2, arg.value.ndarray.memory));
        if (devalloc->alloc_id + 1 == 0) {
          TI_WARN(
              "ignored attempt to launch kernel with ndarray memory of null "
              "handle");
          return;
        }
        const TiNdArray &ndarray = arg.value.ndarray;

        std::vector<int> shape(ndarray.shape.dims,
                               ndarray.shape.dims + ndarray.shape.dim_count);

        if (ndarray.elem_shape.dim_count != 0) {
          std::vector<int> elem_shape(
              ndarray.elem_shape.dims,
              ndarray.elem_shape.dims + ndarray.elem_shape.dim_count);

          runtime_context.set_arg_devalloc(i, *devalloc, shape, elem_shape);
        } else {
          runtime_context.set_arg_devalloc(i, *devalloc, shape);
        }

        devallocs.emplace_back(std::move(devalloc));
        break;
      }
      default:
        TI_ASSERT(false);
    }
  }
  ((taichi::lang::aot::Kernel *)kernel)->launch(&runtime_context);
}

void ti_launch_compute_graph(TiRuntime runtime,
                             TiComputeGraph compute_graph,
                             uint32_t arg_count,
                             const TiNamedArgument *args) {
  if (runtime == nullptr) {
    TI_WARN(
        "ignored attempt to launch compute graph on runtime of null handle");
    return;
  }
  if (compute_graph == nullptr) {
    TI_WARN("ignored attempt to launch compute graph of null handle");
    return;
  }

  Runtime &runtime2 = *((Runtime *)runtime);
  std::unordered_map<std::string, taichi::lang::aot::IValue> arg_map{};
  std::vector<taichi::lang::Ndarray> ndarrays;
  ndarrays.reserve(arg_count);

  for (uint32_t i = 0; i < arg_count; ++i) {
    const auto &arg = args[i];
    switch (arg.argument.type) {
      case TI_ARGUMENT_TYPE_I32: {
        arg_map.emplace(
            std::make_pair(arg.name, taichi::lang::aot::IValue::create<int32_t>(
                                         arg.argument.value.i32)));
        break;
      }
      case TI_ARGUMENT_TYPE_F32: {
        arg_map.emplace(std::make_pair(
            arg.name,
            taichi::lang::aot::IValue::create<float>(arg.argument.value.f32)));
        break;
      }
      case TI_ARGUMENT_TYPE_NDARRAY: {
        taichi::lang::DeviceAllocation devalloc =
            devmem2devalloc(runtime2, arg.argument.value.ndarray.memory);
        if (devalloc.alloc_id + 1 == 0) {
          TI_WARN(
              "ignored attempt to launch kernel with ndarray memory of null "
              "handle");
          return;
        }
        const TiNdArray &ndarray = arg.argument.value.ndarray;

        std::vector<int> shape(ndarray.shape.dims,
                               ndarray.shape.dims + ndarray.shape.dim_count);

        std::vector<int> elem_shape(
            ndarray.elem_shape.dims,
            ndarray.elem_shape.dims + ndarray.elem_shape.dim_count);

        ndarrays.emplace_back(taichi::lang::Ndarray(
            devalloc, taichi::lang::PrimitiveType::f32, shape, elem_shape));
        arg_map.emplace(std::make_pair(
            arg.name, taichi::lang::aot::IValue::create(ndarrays.back())));
        break;
      }
      default:
        TI_ASSERT(false);
    }
  }
  ((taichi::lang::aot::CompiledGraph *)compute_graph)->run(arg_map);
}
void ti_submit(TiRuntime runtime) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to submit to runtime of null handle");
    return;
  }

  ((Runtime *)runtime)->submit();
}
void ti_wait(TiRuntime runtime) {
  if (runtime == nullptr) {
    TI_WARN("ignored attempt to wait on runtime of null handle");
    return;
  }

  ((Runtime *)runtime)->wait();
}
