#include "taichi_core_impl.h"
#include "taichi_opengl_impl.h"
#include "taichi_vulkan_impl.h"
#include "taichi_llvm_impl.h"
#include "taichi/program/ndarray.h"

struct ErrorCache {
  TiError error{TI_ERROR_SUCCESS};
  std::string message{};
};

namespace {
// Error is recorded on a per-thread basis.
thread_local ErrorCache thread_error_cache;

const char *describe_error(TiError error) {
  switch (error) {
    case TI_ERROR_INCOMPLETE:
      return "incomplete";
    case TI_ERROR_SUCCESS:
      return "success";
    case TI_ERROR_NOT_SUPPORTED:
      return "not supported";
    case TI_ERROR_CORRUPTED_DATA:
      return "path not found";
    case TI_ERROR_NAME_NOT_FOUND:
      return "name not found";
    case TI_ERROR_INVALID_ARGUMENT:
      return "invalid argument";
    case TI_ERROR_ARGUMENT_NULL:
      return "argument null";
    case TI_ERROR_ARGUMENT_OUT_OF_RANGE:
      return "argument out of range";
    case TI_ERROR_ARGUMENT_NOT_FOUND:
      return "argument not found";
    case TI_ERROR_INVALID_INTEROP:
      return "invalid interop";
    default:
      return "unknown error";
  }
}
}  // namespace

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

TiMemory Runtime::allocate_memory(
    const taichi::lang::Device::AllocParams &params) {
  taichi::lang::DeviceAllocation devalloc = this->get().allocate_memory(params);
  return devalloc2devmem(*this, devalloc);
}
void Runtime::free_memory(TiMemory devmem) {
  this->get().dealloc_memory(devmem2devalloc(*this, devmem));
}

AotModule::AotModule(Runtime &runtime,
                     std::unique_ptr<taichi::lang::aot::Module> aot_module)
    : runtime_(&runtime), aot_module_(std::move(aot_module)) {
}

taichi::lang::aot::Kernel *AotModule::get_kernel(const std::string &name) {
  return aot_module_->get_kernel(name);
}
taichi::lang::aot::CompiledGraph *AotModule::get_cgraph(
    const std::string &name) {
  auto it = loaded_cgraphs_.find(name);
  if (it == loaded_cgraphs_.end()) {
    return loaded_cgraphs_
        .emplace(std::make_pair(name, aot_module_->get_graph(name)))
        .first->second.get();
  } else {
    return it->second.get();
  }
}
taichi::lang::aot::Module &AotModule::get() {
  return *aot_module_;
}
Runtime &AotModule::runtime() {
  return *runtime_;
}

Event::Event(Runtime &runtime, std::unique_ptr<taichi::lang::DeviceEvent> event)
    : runtime_(&runtime), event_(std::move(event)) {
}

taichi::lang::DeviceEvent &Event::get() {
  return *event_;
}
Runtime &Event::runtime() {
  return *runtime_;
}

// -----------------------------------------------------------------------------

TiError ti_get_last_error(uint64_t message_size, char *message) {
  // Emit message only if the output buffer is property provided.
  if (message_size > 0 && message != nullptr) {
    size_t n = thread_error_cache.message.size();
    if (n >= message_size) {
      n = message_size - 1;  // -1 for the byte of `\0`.
    }
    std::memcpy(message, thread_error_cache.message.data(), n);
    message[n] = '\0';
  }
  return thread_error_cache.error;
}
// C-API errors MUST be set via this interface. No matter from internal or
// external procedures.
void ti_set_last_error(TiError error, const char *message) {
  if (error < TI_ERROR_SUCCESS) {
    TI_WARN("C-API error: ({}) {}", describe_error(error), message);
    if (message != nullptr) {
      thread_error_cache.message = message;
    } else {
      thread_error_cache.message.clear();
    }
    thread_error_cache.error = error;
  } else {
    thread_error_cache.error = TI_ERROR_SUCCESS;
    thread_error_cache.message.clear();
  }
}

TiRuntime ti_create_runtime(TiArch arch) {
  switch (arch) {
#ifdef TI_WITH_VULKAN
    case TI_ARCH_VULKAN: {
      return (TiRuntime)(static_cast<Runtime *>(new VulkanRuntimeOwned));
    }
#endif  // TI_WITH_VULKAN
#ifdef TI_WITH_OPENGL
    case TI_ARCH_OPENGL: {
      return (TiRuntime)(static_cast<Runtime *>(new OpenglRuntime));
    }
#endif  // TI_WITH_OPENGL
#ifdef TI_WITH_LLVM
    case TI_ARCH_X64: {
      return (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::x64)));
    }
    case TI_ARCH_ARM64: {
      return (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::arm64)));
    }
    case TI_ARCH_CUDA: {
      return (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::cuda)));
    }
#endif  // TI_WITH_LLVM
    default: {
      TI_CAPI_NOT_SUPPORTED(arch);
      return TI_NULL_HANDLE;
    }
  }
  return TI_NULL_HANDLE;
}
void ti_destroy_runtime(TiRuntime runtime) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  delete (Runtime *)runtime;
}

TiMemory ti_allocate_memory(TiRuntime runtime,
                            const TiMemoryAllocateInfo *create_info) {
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(create_info);

  taichi::lang::AllocUsage usage{};
  if (create_info->usage & TI_MEMORY_USAGE_STORAGE_BIT) {
    usage = usage | taichi::lang::AllocUsage::Storage;
  }
  if (create_info->usage & TI_MEMORY_USAGE_UNIFORM_BIT) {
    usage = usage | taichi::lang::AllocUsage::Uniform;
  }
  if (create_info->usage & TI_MEMORY_USAGE_VERTEX_BIT) {
    usage = usage | taichi::lang::AllocUsage::Vertex;
  }
  if (create_info->usage & TI_MEMORY_USAGE_INDEX_BIT) {
    usage = usage | taichi::lang::AllocUsage::Index;
  }

  taichi::lang::Device::AllocParams params{};
  params.size = create_info->size;
  params.host_write = create_info->host_write;
  params.host_read = create_info->host_read;
  params.export_sharing = create_info->export_sharing;
  params.usage = usage;

  TiMemory devmem = ((Runtime *)runtime)->allocate_memory(params);
  return devmem;
}

void ti_free_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->free_memory(devmem);
}

void *ti_map_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  return runtime2->get().map(devmem2devalloc(*runtime2, devmem));
}
void ti_unmap_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->get().unmap(devmem2devalloc(*runtime2, devmem));
}

TiTexture ti_allocate_texture(TiRuntime runtime,
                              const TiTextureAllocateInfo *allocate_info) {
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(allocate_info);

  TI_CAPI_NOT_SUPPORTED_IF_RV(allocate_info->mip_level_count > 1);
  TI_CAPI_NOT_SUPPORTED_IF_RV(allocate_info->extent.array_layer_count > 1);

  taichi::lang::ImageAllocUsage usage{};
  if (allocate_info->usage & TI_TEXTURE_USAGE_STORAGE_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Storage;
  }
  if (allocate_info->usage & TI_TEXTURE_USAGE_SAMPLED_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Sampled;
  }
  if (allocate_info->usage & TI_TEXTURE_USAGE_ATTACHMENT_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Attachment;
  }

  switch ((taichi::lang::ImageDimension)allocate_info->dimension) {
#define PER_IMAGE_DIMENSION(x) case taichi::lang::ImageDimension::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_DIMENSION
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "allocate_info->dimension");
      return TI_NULL_HANDLE;
    }
  }

  switch ((taichi::lang::BufferFormat)allocate_info->format) {
#define PER_BUFFER_FORMAT(x) case taichi::lang::BufferFormat::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_BUFFER_FORMAT
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                        "allocate_info->format");
      return TI_NULL_HANDLE;
    }
  }

  taichi::lang::ImageParams params{};
  params.x = allocate_info->extent.width;
  params.y = allocate_info->extent.height;
  params.z = allocate_info->extent.depth;
  params.dimension = (taichi::lang::ImageDimension)allocate_info->dimension;
  params.format = (taichi::lang::BufferFormat)allocate_info->format;
  params.export_sharing = false;
  params.usage = usage;

  TiTexture devtex = ((Runtime *)runtime)->allocate_texture(params);
  return devtex;
}
void ti_free_texture(TiRuntime runtime, TiTexture texture) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(texture);

  ((Runtime *)runtime)->free_texture(texture);
}

TiEvent ti_create_event(TiRuntime runtime) {
  TI_CAPI_ARGUMENT_NULL_RV(runtime);

  Runtime *runtime2 = (Runtime *)runtime;
  std::unique_ptr<taichi::lang::DeviceEvent> event =
      runtime2->get().create_event();
  Event *event2 = new Event(*runtime2, std::move(event));
  return (TiEvent)event2;
}
void ti_destroy_event(TiEvent event) {
  TI_CAPI_ARGUMENT_NULL(event);

  delete (Event *)event;
}

void ti_copy_memory_device_to_device(TiRuntime runtime,
                                     const TiMemorySlice *dst_memory,
                                     const TiMemorySlice *src_memory) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_memory);
  TI_CAPI_ARGUMENT_NULL(dst_memory->memory);
  TI_CAPI_ARGUMENT_NULL(src_memory);
  TI_CAPI_ARGUMENT_NULL(src_memory->memory);
  TI_CAPI_INVALID_ARGUMENT(dst_memory->memory != src_memory->memory);

  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = devmem2devalloc(*runtime2, dst_memory->memory)
                 .get_ptr(dst_memory->offset);
  auto src = devmem2devalloc(*runtime2, src_memory->memory)
                 .get_ptr(src_memory->offset);
  runtime2->buffer_copy(dst, src, dst_memory->size);
}

void ti_copy_texture_device_to_device(TiRuntime runtime,
                                      const TiTextureSlice *dst_texture,
                                      const TiTextureSlice *src_texture) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_texture);
  TI_CAPI_ARGUMENT_NULL(dst_texture->texture);
  TI_CAPI_ARGUMENT_NULL(src_texture);
  TI_CAPI_ARGUMENT_NULL(src_texture->texture);
  TI_CAPI_INVALID_ARGUMENT(src_texture->extent.width !=
                           dst_texture->extent.width);
  TI_CAPI_INVALID_ARGUMENT(src_texture->extent.height !=
                           dst_texture->extent.height);
  TI_CAPI_INVALID_ARGUMENT(src_texture->extent.depth !=
                           dst_texture->extent.depth);
  TI_CAPI_INVALID_ARGUMENT(src_texture->extent.array_layer_count !=
                           dst_texture->extent.array_layer_count);

  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = devtex2devalloc(*runtime2, dst_texture->texture);
  auto src = devtex2devalloc(*runtime2, src_texture->texture);

  taichi::lang::ImageCopyParams params{};
  params.width = dst_texture->extent.width;
  params.height = dst_texture->extent.height;
  params.depth = dst_texture->extent.depth;
  runtime2->copy_image(dst, src, params);
}
void ti_transition_texture(TiRuntime runtime,
                           TiTexture texture,
                           TiTextureLayout layout) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(texture);

  Runtime *runtime2 = (Runtime *)runtime;
  auto image = devtex2devalloc(*runtime2, texture);
  auto layout2 = (taichi::lang::ImageLayout)layout;

  switch ((taichi::lang::ImageLayout)layout) {
#define PER_IMAGE_LAYOUT(x) case taichi::lang::ImageLayout::x:
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_LAYOUT
    break;
    default: {
      ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "layout");
      return;
    }
  }

  runtime2->transition_image(image, layout2);
}

TiAotModule ti_load_aot_module(TiRuntime runtime, const char *module_path) {
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module_path);

  TiAotModule aot_module = ((Runtime *)runtime)->load_aot_module(module_path);

  if (aot_module == TI_NULL_HANDLE) {
    ti_set_last_error(TI_ERROR_CORRUPTED_DATA, module_path);
    return TI_NULL_HANDLE;
  }
  return aot_module;
}
void ti_destroy_aot_module(TiAotModule aot_module) {
  TI_CAPI_ARGUMENT_NULL(aot_module);

  delete (AotModule *)aot_module;
}

TiKernel ti_get_aot_module_kernel(TiAotModule aot_module, const char *name) {
  TI_CAPI_ARGUMENT_NULL_RV(aot_module);
  TI_CAPI_ARGUMENT_NULL_RV(name);

  taichi::lang::aot::Kernel *kernel =
      ((AotModule *)aot_module)->get_kernel(name);

  if (kernel == nullptr) {
    ti_set_last_error(TI_ERROR_NAME_NOT_FOUND, name);
    return TI_NULL_HANDLE;
  }

  return (TiKernel)kernel;
}

TiComputeGraph ti_get_aot_module_compute_graph(TiAotModule aot_module,
                                               const char *name) {
  TI_CAPI_ARGUMENT_NULL_RV(aot_module);
  TI_CAPI_ARGUMENT_NULL_RV(name);

  taichi::lang::aot::CompiledGraph *cgraph =
      ((AotModule *)aot_module)->get_cgraph(name);

  if (cgraph == nullptr) {
    ti_set_last_error(TI_ERROR_NAME_NOT_FOUND, name);
    return TI_NULL_HANDLE;
  }

  return (TiComputeGraph)cgraph;
}

void ti_launch_kernel(TiRuntime runtime,
                      TiKernel kernel,
                      uint32_t arg_count,
                      const TiArgument *args) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(kernel);
  if (arg_count > 0) {
    TI_CAPI_ARGUMENT_NULL(args);
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
        TI_CAPI_ARGUMENT_NULL(args[i].value.ndarray.memory);

        // Don't allocate it on stack. `DeviceAllocation` is referred to by
        // `GfxRuntime::launch_kernel`.
        std::unique_ptr<taichi::lang::DeviceAllocation> devalloc =
            std::make_unique<taichi::lang::DeviceAllocation>(
                devmem2devalloc(runtime2, arg.value.ndarray.memory));
        const TiNdArray &ndarray = arg.value.ndarray;

        std::vector<int> shape(ndarray.shape.dims,
                               ndarray.shape.dims + ndarray.shape.dim_count);

        runtime_context.set_arg_ndarray(i, (intptr_t)devalloc.get(), shape);

        devallocs.emplace_back(std::move(devalloc));
        break;
      }
      default: {
        ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                          ("args[" + std::to_string(i) + "].type").c_str());
        return;
      }
    }
  }
  ((taichi::lang::aot::Kernel *)kernel)->launch(&runtime_context);
}

void ti_launch_compute_graph(TiRuntime runtime,
                             TiComputeGraph compute_graph,
                             uint32_t arg_count,
                             const TiNamedArgument *args) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(compute_graph);
  if (arg_count > 0) {
    TI_CAPI_ARGUMENT_NULL(args);
  }

  Runtime &runtime2 = *((Runtime *)runtime);
  std::unordered_map<std::string, taichi::lang::aot::IValue> arg_map{};
  std::vector<taichi::lang::Ndarray> ndarrays;
  ndarrays.reserve(arg_count);

  for (uint32_t i = 0; i < arg_count; ++i) {
    TI_CAPI_ARGUMENT_NULL(args[i].name);

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
        TI_CAPI_ARGUMENT_NULL(args[i].argument.value.ndarray.memory);

        taichi::lang::DeviceAllocation devalloc =
            devmem2devalloc(runtime2, arg.argument.value.ndarray.memory);
        const TiNdArray &ndarray = arg.argument.value.ndarray;

        std::vector<int> shape(ndarray.shape.dims,
                               ndarray.shape.dims + ndarray.shape.dim_count);

        std::vector<int> elem_shape(
            ndarray.elem_shape.dims,
            ndarray.elem_shape.dims + ndarray.elem_shape.dim_count);

        const taichi::lang::DataType *prim_ty;
        switch (ndarray.elem_type) {
          case TI_DATA_TYPE_F16:
            prim_ty = &taichi::lang::PrimitiveType::f16;
            break;
          case TI_DATA_TYPE_F32:
            prim_ty = &taichi::lang::PrimitiveType::f32;
            break;
          case TI_DATA_TYPE_F64:
            prim_ty = &taichi::lang::PrimitiveType::f64;
            break;
          case TI_DATA_TYPE_I8:
            prim_ty = &taichi::lang::PrimitiveType::i8;
            break;
          case TI_DATA_TYPE_I16:
            prim_ty = &taichi::lang::PrimitiveType::i16;
            break;
          case TI_DATA_TYPE_I32:
            prim_ty = &taichi::lang::PrimitiveType::i32;
            break;
          case TI_DATA_TYPE_I64:
            prim_ty = &taichi::lang::PrimitiveType::i64;
            break;
          case TI_DATA_TYPE_U8:
            prim_ty = &taichi::lang::PrimitiveType::u8;
            break;
          case TI_DATA_TYPE_U16:
            prim_ty = &taichi::lang::PrimitiveType::u16;
            break;
          case TI_DATA_TYPE_U32:
            prim_ty = &taichi::lang::PrimitiveType::u32;
            break;
          case TI_DATA_TYPE_U64:
            prim_ty = &taichi::lang::PrimitiveType::u64;
            break;
          case TI_DATA_TYPE_GEN:
            prim_ty = &taichi::lang::PrimitiveType::gen;
            break;
          default: {
            ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                              ("args[" + std::to_string(i) +
                               "].argument.value.ndarray.elem_type")
                                  .c_str());
            return;
          }
        }

        taichi::lang::DataType dtype = *prim_ty;
        if (elem_shape.size() > 0) {
          dtype = taichi::lang::TypeFactory::get_instance().get_tensor_type(
              elem_shape, dtype);
        }
        ndarrays.emplace_back(taichi::lang::Ndarray(devalloc, dtype, shape));
        arg_map.emplace(std::make_pair(
            arg.name, taichi::lang::aot::IValue::create(ndarrays.back())));
        break;
      }
      default: {
        ti_set_last_error(
            TI_ERROR_ARGUMENT_OUT_OF_RANGE,
            ("args[" + std::to_string(i) + "].argument.type").c_str());
        return;
      }
    }
  }
  ((taichi::lang::aot::CompiledGraph *)compute_graph)->run(arg_map);
}

void ti_signal_event(TiRuntime runtime, TiEvent event) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(event);

  ((Runtime *)runtime)->signal_event(&((Event *)event)->get());
}

void ti_reset_event(TiRuntime runtime, TiEvent event) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(event);

  ((Runtime *)runtime)->reset_event(&((Event *)event)->get());
}

void ti_wait_event(TiRuntime runtime, TiEvent event) {
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(event);

  ((Runtime *)runtime)->wait_event(&((Event *)event)->get());
}

void ti_submit(TiRuntime runtime) {
  TI_CAPI_ARGUMENT_NULL(runtime);

  ((Runtime *)runtime)->submit();
}
void ti_wait(TiRuntime runtime) {
  TI_CAPI_ARGUMENT_NULL(runtime);

  ((Runtime *)runtime)->wait();
}
