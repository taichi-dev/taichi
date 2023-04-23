#include "taichi_core_impl.h"
#include "taichi_opengl_impl.h"
#include "taichi_vulkan_impl.h"
#include "taichi_llvm_impl.h"
#include "taichi_metal_impl.h"
#include "taichi/program/ndarray.h"
#include "taichi/program/texture.h"
#include "taichi/program/launch_context_builder.h"
#include "taichi/common/virtual_dir.h"
#include "taichi/common/utils.h"

bool is_vulkan_available() {
#ifdef TI_WITH_VULKAN
  return taichi::lang::vulkan::is_vulkan_api_available();
#else
  return false;
#endif
}

bool is_opengl_available() {
#ifdef TI_WITH_OPENGL
  return taichi::lang::opengl::is_opengl_api_available();
#else
  return false;
#endif
}

bool is_cuda_available() {
#ifdef TI_WITH_CUDA
  return taichi::is_cuda_api_available();
#else
  return false;
#endif
}

bool is_x64_available() {
#if defined(TI_WITH_LLVM) &&                                           \
    (defined(__x86_64__) || defined(__x86_64) || defined(__amd64__) || \
     defined(__amd64) || defined(_M_X64))
  return true;
#else
  return false;
#endif
}

bool is_arm64_available() {
#if defined(TI_WITH_LLVM) && (defined(__arm64__) || defined(__aarch64__))
  return true;
#else
  return false;
#endif
}

bool is_metal_available() {
#if defined(__APPLE__) && defined(TI_WITH_METAL)
  return taichi::lang::metal::is_metal_api_available();
#else
  return false;
#endif  // __APPLE__
}

struct ErrorCache {
  TiError error{TI_ERROR_SUCCESS};
  std::string message{};
};

namespace {
// Error is recorded on a per-thread basis.
thread_local ErrorCache thread_error_cache;

const char *describe_error(TiError error) {
  switch (error) {
    case TI_ERROR_SUCCESS:
      return "success";
    case TI_ERROR_NOT_SUPPORTED:
      return "not supported";
    case TI_ERROR_CORRUPTED_DATA:
      return "corrupted data";
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
    case TI_ERROR_INVALID_STATE:
      return "invalid state";
    case TI_ERROR_INCOMPATIBLE_MODULE:
      return "incompatible module";
    case TI_ERROR_OUT_OF_MEMORY:
      return "out of memory";
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
capi::MetalRuntime *Runtime::as_mtl() {
  TI_ASSERT(arch == taichi::Arch::metal);
#ifdef TI_WITH_METAL
  return static_cast<capi::MetalRuntime *>(this);
#else
  return nullptr;
#endif
}

TiMemory Runtime::allocate_memory(
    const taichi::lang::Device::AllocParams &params) {
  taichi::lang::DeviceAllocation devalloc;
  taichi::lang::RhiResult res = this->get().allocate_memory(params, &devalloc);
  if (res != taichi::lang::RhiResult::success) {
    throw std::bad_alloc();
  }
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

// -----------------------------------------------------------------------------

uint32_t ti_get_version() {
  return TI_C_API_VERSION;
}

void ti_get_available_archs(uint32_t *arch_count, TiArch *archs) {
  if (arch_count == nullptr) {
    return;
  }

  thread_local std::vector<TiArch> AVAILABLE_ARCHS{};
  if (AVAILABLE_ARCHS.empty()) {
    if (is_vulkan_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_VULKAN);
    }
    if (is_metal_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_METAL);
    }
    if (is_cuda_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_CUDA);
    }
    if (is_x64_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_X64);
    }
    if (is_arm64_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_ARM64);
    }
    if (is_opengl_available()) {
      AVAILABLE_ARCHS.emplace_back(TI_ARCH_OPENGL);
    }
  }

  size_t n = std::min((size_t)*arch_count, AVAILABLE_ARCHS.size());
  *arch_count = (uint32_t)AVAILABLE_ARCHS.size();
  if (archs != nullptr) {
    for (size_t i = 0; i < n; ++i) {
      archs[i] = AVAILABLE_ARCHS.at(i);
    }
  }
}

TiError ti_get_last_error(uint64_t *message_size, char *message) {
  TiError out = TI_ERROR_INVALID_STATE;
  TI_CAPI_TRY_CATCH_BEGIN();
  out = thread_error_cache.error;

  // Emit message only if the output buffer is property provided.
  if (message_size == nullptr) {
    return out;
  }
  size_t buffer_size = *message_size;
  *message_size = thread_error_cache.message.size() + 1;

  if (buffer_size > 0 && message != nullptr) {
    // -1 for the byte of `\0`.
    size_t n = std::min(thread_error_cache.message.size(), buffer_size - 1);
    std::memcpy(message, thread_error_cache.message.data(), n);
    message[n] = '\0';
  }
  TI_CAPI_TRY_CATCH_END();
  return out;
}
// C-API errors MUST be set via this interface. No matter from internal or
// external procedures.
void ti_set_last_error(TiError error, const char *message) {
  TI_CAPI_TRY_CATCH_BEGIN();
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
  TI_CAPI_TRY_CATCH_END();
}

TiRuntime ti_create_runtime(TiArch arch, uint32_t device_index) {
  TiRuntime out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  // FIXME: (penguinliong) Support device selection.
  TI_CAPI_NOT_SUPPORTED_IF_RV(device_index != 0);
  TI_INFO("Taichi Runtime C-API version is: {}", TI_C_API_VERSION);
  switch (arch) {
#ifdef TI_WITH_VULKAN
    case TI_ARCH_VULKAN: {
      VulkanRuntimeOwned *vulkan_runtime;
      if (is_ci()) {
        auto param = make_vulkan_runtime_creator_params();
        param.enable_validation_layer = true;
        vulkan_runtime = new VulkanRuntimeOwned(std::move(param));
      } else {
        vulkan_runtime = new VulkanRuntimeOwned;
      }
      out = (TiRuntime)(static_cast<Runtime *>(vulkan_runtime));
      break;
    }
#endif  // TI_WITH_VULKAN
#ifdef TI_WITH_OPENGL
    case TI_ARCH_OPENGL: {
      out = (TiRuntime)(static_cast<Runtime *>(new OpenglRuntime));
      break;
    }
#endif  // TI_WITH_OPENGL
#ifdef TI_WITH_LLVM
    case TI_ARCH_X64: {
      out = (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::x64)));
      break;
    }
    case TI_ARCH_ARM64: {
      out = (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::arm64)));
      break;
    }
    case TI_ARCH_CUDA: {
      out = (TiRuntime)(static_cast<Runtime *>(
          new capi::LlvmRuntime(taichi::Arch::cuda)));
      break;
    }
#endif  // TI_WITH_LLVM
#ifdef TI_WITH_METAL
    case TI_ARCH_METAL: {
      out = (TiRuntime)(static_cast<Runtime *>(new capi::MetalRuntime()));
      break;
    }
#endif  // TI_WITH_METAL
    default: {
      TI_CAPI_NOT_SUPPORTED(arch);
      return TI_NULL_HANDLE;
    }
  }
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_destroy_runtime(TiRuntime runtime) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  delete (Runtime *)runtime;
  TI_CAPI_TRY_CATCH_END();
}

void ti_set_runtime_capabilities_ext(
    TiRuntime runtime,
    uint32_t capability_count,
    const TiCapabilityLevelInfo *capabilities) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  Runtime *runtime2 = (Runtime *)runtime;
  taichi::lang::DeviceCapabilityConfig devcaps;
  for (uint32_t i = 0; i < capability_count; ++i) {
    const auto &cap_level_info = capabilities[i];
    devcaps.set((taichi::lang::DeviceCapability)cap_level_info.capability,
                cap_level_info.level);
  }
  runtime2->get().set_caps(std::move(devcaps));

  TI_CAPI_TRY_CATCH_END();
}

void ti_get_runtime_capabilities(TiRuntime runtime,
                                 uint32_t *capability_count,
                                 TiCapabilityLevelInfo *capabilities) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  Runtime *runtime2 = (Runtime *)runtime;
  const taichi::lang::DeviceCapabilityConfig &devcaps =
      runtime2->get().get_caps();

  if (capability_count == nullptr) {
    return;
  }

  if (capabilities != nullptr) {
    auto pos = devcaps.to_inner().begin();
    auto end = devcaps.to_inner().end();
    for (size_t i = 0; i < *capability_count; ++i) {
      if (pos == end) {
        break;
      }
      capabilities[i].capability = (TiCapability)(uint32_t)pos->first;
      capabilities[i].level = pos->second;
      ++pos;
    }
  }

  *capability_count = devcaps.to_inner().size();

  TI_CAPI_TRY_CATCH_END();
}

TiMemory ti_allocate_memory(TiRuntime runtime,
                            const TiMemoryAllocateInfo *create_info) {
  TiMemory out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
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

  try {
    out = ((Runtime *)runtime)->allocate_memory(params);
  } catch (const std::bad_alloc &e) {
    ti_set_last_error(TI_ERROR_OUT_OF_MEMORY, "allocate_memory");
    return TI_NULL_HANDLE;
  }
  TI_CAPI_TRY_CATCH_END();
  return out;
}

void ti_free_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->free_memory(devmem);
  TI_CAPI_TRY_CATCH_END();
}

void *ti_map_memory(TiRuntime runtime, TiMemory devmem) {
  void *out = nullptr;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  TI_ASSERT(runtime2->get().map(devmem2devalloc(*runtime2, devmem), &out) ==
                taichi::lang::RhiResult::success &&
            "RHI map memory failed: "
            "Mapping Memory without VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT set");
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_unmap_memory(TiRuntime runtime, TiMemory devmem) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(devmem);

  Runtime *runtime2 = (Runtime *)runtime;
  runtime2->get().unmap(devmem2devalloc(*runtime2, devmem));
  TI_CAPI_TRY_CATCH_END();
}

TiImage ti_allocate_image(TiRuntime runtime,
                          const TiImageAllocateInfo *allocate_info) {
  TiImage out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(allocate_info);

  TI_CAPI_NOT_SUPPORTED_IF_RV(allocate_info->mip_level_count > 1);
  TI_CAPI_NOT_SUPPORTED_IF_RV(allocate_info->extent.array_layer_count > 1);

  taichi::lang::ImageAllocUsage usage{};
  if (allocate_info->usage & TI_IMAGE_USAGE_STORAGE_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Storage;
  }
  if (allocate_info->usage & TI_IMAGE_USAGE_SAMPLED_BIT) {
    usage = usage | taichi::lang::ImageAllocUsage::Sampled;
  }
  if (allocate_info->usage & TI_IMAGE_USAGE_ATTACHMENT_BIT) {
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

  out = ((Runtime *)runtime)->allocate_image(params);
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_free_image(TiRuntime runtime, TiImage image) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(image);

  ((Runtime *)runtime)->free_image(image);
  TI_CAPI_TRY_CATCH_END();
}

TiSampler ti_create_sampler(TiRuntime runtime,
                            const TiSamplerCreateInfo *create_info) {
  TiSampler out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_NOT_SUPPORTED(ti_create_sampler);
  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_destroy_sampler(TiRuntime runtime, TiSampler sampler) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_NOT_SUPPORTED(ti_destroy_sampler);
  TI_CAPI_TRY_CATCH_END();
}

void ti_copy_memory_device_to_device(TiRuntime runtime,
                                     const TiMemorySlice *dst_memory,
                                     const TiMemorySlice *src_memory) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_memory);
  TI_CAPI_ARGUMENT_NULL(dst_memory->memory);
  TI_CAPI_ARGUMENT_NULL(src_memory);
  TI_CAPI_ARGUMENT_NULL(src_memory->memory);
  if (dst_memory->size != src_memory->size) {
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT,
                      "The size of memory slices are not match");
    return;
  }

  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = devmem2devalloc(*runtime2, dst_memory->memory)
                 .get_ptr(dst_memory->offset);
  auto src = devmem2devalloc(*runtime2, src_memory->memory)
                 .get_ptr(src_memory->offset);
  runtime2->buffer_copy(dst, src, dst_memory->size);
  TI_CAPI_TRY_CATCH_END();
}

void ti_copy_image_device_to_device(TiRuntime runtime,
                                    const TiImageSlice *dst_image,
                                    const TiImageSlice *src_image) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(dst_image);
  TI_CAPI_ARGUMENT_NULL(dst_image->image);
  TI_CAPI_ARGUMENT_NULL(src_image);
  TI_CAPI_ARGUMENT_NULL(src_image->image);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.width != dst_image->extent.width);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.height !=
                           dst_image->extent.height);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.depth != dst_image->extent.depth);
  TI_CAPI_INVALID_ARGUMENT(src_image->extent.array_layer_count !=
                           dst_image->extent.array_layer_count);

  Runtime *runtime2 = (Runtime *)runtime;
  auto dst = devimg2devalloc(*runtime2, dst_image->image);
  auto src = devimg2devalloc(*runtime2, src_image->image);

  taichi::lang::ImageCopyParams params{};
  params.width = dst_image->extent.width;
  params.height = dst_image->extent.height;
  params.depth = dst_image->extent.depth;
  runtime2->copy_image(dst, src, params);
  TI_CAPI_TRY_CATCH_END();
}
void ti_track_image_ext(TiRuntime runtime,
                        TiImage image,
                        TiImageLayout layout) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(image);

  Runtime *runtime2 = (Runtime *)runtime;
  auto image2 = devimg2devalloc(*runtime2, image);
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

  runtime2->track_image(image2, layout2);
  TI_CAPI_TRY_CATCH_END();
}
void ti_transition_image(TiRuntime runtime,
                         TiImage texture,
                         TiImageLayout layout) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(texture);

  Runtime *runtime2 = (Runtime *)runtime;
  auto image = devimg2devalloc(*runtime2, texture);
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
  TI_CAPI_TRY_CATCH_END();
}

TiAotModule ti_load_aot_module(TiRuntime runtime, const char *module_path) {
  TiAotModule out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(module_path);

  // (penguinliong) Should call `create_aot_module` directly after all backends
  // adapted to it.
  TiAotModule aot_module = ((Runtime *)runtime)->load_aot_module(module_path);

  if (aot_module == TI_NULL_HANDLE) {
    ti_set_last_error(TI_ERROR_CORRUPTED_DATA, module_path);
    return TI_NULL_HANDLE;
  }
  out = aot_module;
  TI_CAPI_TRY_CATCH_END();
  return out;
}
TiAotModule ti_create_aot_module(TiRuntime runtime,
                                 const void *tcm,
                                 uint64_t size) {
  TiAotModule out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(runtime);
  TI_CAPI_ARGUMENT_NULL_RV(tcm);

  auto dir = taichi::io::VirtualDir::from_zip(tcm, size);
  if (dir == TI_NULL_HANDLE) {
    ti_set_last_error(TI_ERROR_CORRUPTED_DATA, "tcm");
    return TI_NULL_HANDLE;
  }

  Error err = ((Runtime *)runtime)->create_aot_module(dir.get(), out);
  err.set_last_error();

  TI_CAPI_TRY_CATCH_END();
  return out;
}
void ti_destroy_aot_module(TiAotModule aot_module) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(aot_module);

  delete (AotModule *)aot_module;
  TI_CAPI_TRY_CATCH_END();
}

TiKernel ti_get_aot_module_kernel(TiAotModule aot_module, const char *name) {
  TiKernel out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(aot_module);
  TI_CAPI_ARGUMENT_NULL_RV(name);

  taichi::lang::aot::Kernel *kernel =
      ((AotModule *)aot_module)->get_kernel(name);

  if (kernel == nullptr) {
    ti_set_last_error(TI_ERROR_NAME_NOT_FOUND, name);
    return TI_NULL_HANDLE;
  }

  out = (TiKernel)kernel;
  TI_CAPI_TRY_CATCH_END();
  return out;
}

TiComputeGraph ti_get_aot_module_compute_graph(TiAotModule aot_module,
                                               const char *name) {
  TiComputeGraph out = TI_NULL_HANDLE;
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL_RV(aot_module);
  TI_CAPI_ARGUMENT_NULL_RV(name);

  taichi::lang::aot::CompiledGraph *cgraph =
      ((AotModule *)aot_module)->get_cgraph(name);

  if (cgraph == nullptr) {
    ti_set_last_error(TI_ERROR_NAME_NOT_FOUND, name);
    return TI_NULL_HANDLE;
  }

  out = (TiComputeGraph)cgraph;
  TI_CAPI_TRY_CATCH_END();
  return out;
}

void ti_launch_kernel(TiRuntime runtime,
                      TiKernel kernel,
                      uint32_t arg_count,
                      const TiArgument *args) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(kernel);
  if (arg_count > 0) {
    TI_CAPI_ARGUMENT_NULL(args);
  }
  if (arg_count > taichi_max_num_args_total) {
    ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE, "arg_count");
    return;
  }

  auto ti_kernel = (taichi::lang::aot::Kernel *)kernel;

  Runtime &runtime2 = *((Runtime *)runtime);
  taichi::lang::LaunchContextBuilder builder(ti_kernel);
  std::vector<std::unique_ptr<taichi::lang::DeviceAllocation>> devallocs;

  for (uint32_t i = 0; i < arg_count; ++i) {
    const auto &arg = args[i];
    switch (arg.type) {
      case TI_ARGUMENT_TYPE_SCALAR: {
        switch (arg.value.scalar.type) {
          case TI_DATA_TYPE_I16: {
            int16_t arg_val;
            std::memcpy(&arg_val, &arg.value.scalar.value.x16, sizeof(arg_val));
            builder.set_arg(i, arg_val);
            break;
          }
          case TI_DATA_TYPE_U16: {
            uint16_t arg_val = arg.value.scalar.value.x16;
            builder.set_arg(i, arg_val);
            break;
          }
          case TI_DATA_TYPE_F16: {
            float arg_val;
            std::memcpy(&arg_val, &arg.value.scalar.value.x32, sizeof(arg_val));
            // FIXME: temporary workaround for f16
            builder.set_arg_float(i, arg_val);
            break;
          }
          default: {
            ti_set_last_error(
                TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                ("args[" + std::to_string(i) + "].value.scalar.type").c_str());
            return;
          }
        }
        break;
      }

      case TI_ARGUMENT_TYPE_I32: {
        builder.set_arg(i, arg.value.i32);
        break;
      }
      case TI_ARGUMENT_TYPE_F32: {
        builder.set_arg(i, arg.value.f32);
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

        builder.set_arg_ndarray_impl(i, (intptr_t)devalloc.get(), shape);

        devallocs.emplace_back(std::move(devalloc));
        break;
      }
      case TI_ARGUMENT_TYPE_TEXTURE: {
        TI_CAPI_ARGUMENT_NULL(args[i].value.texture.image);
        std::unique_ptr<taichi::lang::DeviceAllocation> devalloc =
            std::make_unique<taichi::lang::DeviceAllocation>(
                devimg2devalloc(runtime2, arg.value.texture.image));
        int width = arg.value.texture.extent.width;
        int height = arg.value.texture.extent.height;
        int depth = arg.value.texture.extent.depth;
        builder.set_arg_rw_texture_impl(i, (intptr_t)devalloc.get(),
                                        {width, height, depth});
        devallocs.emplace_back(std::move(devalloc));
        break;
      }
      case TI_ARGUMENT_TYPE_TENSOR: {
        auto &tensor = arg.value.tensor;
        TI_CAPI_ARGUMENT_NULL(tensor.data);
        if (tensor.width == 2) {
          for (int j = 0; j < tensor.num_elements; j++) {
            builder.set_struct_arg_impl(
                {(int)i, j}, taichi_union_cast_with_different_sizes<uint16_t>(
                                 tensor.data[j]));
          }
        } else if (tensor.width == 4) {
          for (int j = 0; j < tensor.num_elements; j++) {
            builder.set_struct_arg_impl(
                {(int)i, j}, taichi_union_cast_with_different_sizes<uint32_t>(
                                 tensor.data[j]));
          }
        } else {
          ti_set_last_error(TI_ERROR_NOT_SUPPORTED,
                            ("args[" + std::to_string(i) + "].type").c_str());
        }
        break;
      }
      default: {
        ti_set_last_error(TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                          ("args[" + std::to_string(i) + "].type").c_str());
        return;
      }
    }
  }
  ti_kernel->launch(builder);
  TI_CAPI_TRY_CATCH_END();
}

void ti_launch_compute_graph(TiRuntime runtime,
                             TiComputeGraph compute_graph,
                             uint32_t arg_count,
                             const TiNamedArgument *args) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);
  TI_CAPI_ARGUMENT_NULL(compute_graph);
  if (arg_count > 0) {
    TI_CAPI_ARGUMENT_NULL(args);
  }

  Runtime &runtime2 = *((Runtime *)runtime);
  std::unordered_map<std::string, taichi::lang::aot::IValue> arg_map{};
  std::vector<taichi::lang::Ndarray> ndarrays;
  ndarrays.reserve(arg_count);
  std::vector<taichi::lang::Texture> textures;
  textures.reserve(arg_count);

  for (uint32_t i = 0; i < arg_count; ++i) {
    TI_CAPI_ARGUMENT_NULL(args[i].name);

    const auto &arg = args[i];
    switch (arg.argument.type) {
      case TI_ARGUMENT_TYPE_SCALAR: {
        switch (arg.argument.value.scalar.type) {
          case TI_DATA_TYPE_I16: {
            int16_t arg_val;
            std::memcpy(&arg_val, &arg.argument.value.scalar.value.x16,
                        sizeof(arg_val));
            arg_map.emplace(std::make_pair(
                arg.name, taichi::lang::aot::IValue::create<int16_t>(arg_val)));
            break;
          }
          case TI_DATA_TYPE_U16: {
            uint16_t arg_val = arg.argument.value.scalar.value.x16;
            arg_map.emplace(std::make_pair(
                arg.name,
                taichi::lang::aot::IValue::create<uint16_t>(arg_val)));
            break;
          }
          case TI_DATA_TYPE_F16: {
            float arg_val;
            std::memcpy(&arg_val, &arg.argument.value.scalar.value.x32,
                        sizeof(arg_val));
            arg_map.emplace(std::make_pair(
                arg.name, taichi::lang::aot::IValue::create<float>(arg_val)));
            break;
          }
          default: {
            ti_set_last_error(
                TI_ERROR_ARGUMENT_OUT_OF_RANGE,
                ("args[" + std::to_string(i) + "].value.scalar.type").c_str());
            return;
          }
        }
        break;
      }
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
      case TI_ARGUMENT_TYPE_TEXTURE: {
        TI_CAPI_ARGUMENT_NULL(args[i].argument.value.texture.image);

        taichi::lang::DeviceAllocation devalloc =
            devimg2devalloc(runtime2, arg.argument.value.texture.image);
        taichi::lang::BufferFormat format =
            (taichi::lang::BufferFormat)arg.argument.value.texture.format;
        uint32_t width = arg.argument.value.texture.extent.width;
        uint32_t height = arg.argument.value.texture.extent.height;
        uint32_t depth = arg.argument.value.texture.extent.depth;

        textures.emplace_back(
            taichi::lang::Texture(devalloc, format, width, height, depth));
        arg_map.emplace(std::make_pair(
            arg.name, taichi::lang::aot::IValue::create(textures.back())));
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
  TI_CAPI_TRY_CATCH_END();
}

void ti_flush(TiRuntime runtime) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  ((Runtime *)runtime)->flush();
  TI_CAPI_TRY_CATCH_END();
}
void ti_wait(TiRuntime runtime) {
  TI_CAPI_TRY_CATCH_BEGIN();
  TI_CAPI_ARGUMENT_NULL(runtime);

  ((Runtime *)runtime)->wait();
  TI_CAPI_TRY_CATCH_END();
}
