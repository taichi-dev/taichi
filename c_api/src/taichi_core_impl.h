#pragma once
#include "taichi/taichi_core.h"
#include "taichi/aot/module_loader.h"
#include "taichi/rhi/device.h"
#include "taichi/program/texture.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

// Error reporting.
#define TI_CAPI_NOT_SUPPORTED(x) ti_set_last_error(TI_ERROR_NOT_SUPPORTED, #x);
#define TI_CAPI_NOT_SUPPORTED_IF(x)                \
  if (x) {                                         \
    ti_set_last_error(TI_ERROR_NOT_SUPPORTED, #x); \
  }
#define TI_CAPI_NOT_SUPPORTED_IF_RV(x)             \
  if (x) {                                         \
    ti_set_last_error(TI_ERROR_NOT_SUPPORTED, #x); \
    return TI_NULL_HANDLE;                         \
  }

#define TI_CAPI_ARGUMENT_NULL(x)                   \
  if (x == TI_NULL_HANDLE) {                       \
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, #x); \
    return;                                        \
  }
#define TI_CAPI_ARGUMENT_NULL_RV(x)                \
  if (x == TI_NULL_HANDLE) {                       \
    ti_set_last_error(TI_ERROR_ARGUMENT_NULL, #x); \
    return TI_NULL_HANDLE;                         \
  }

#define TI_CAPI_INVALID_ARGUMENT(x)                   \
  if (x == TI_NULL_HANDLE) {                          \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, #x); \
    return;                                           \
  }
#define TI_CAPI_INVALID_ARGUMENT_RV(x)                \
  if (x == TI_NULL_HANDLE) {                          \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, #x); \
    return TI_NULL_HANDLE;                            \
  }

#define TI_CAPI_INVALID_INTEROP_ARCH(x, arch)                    \
  if (x != taichi::Arch::arch) {                                 \
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!=" #arch); \
    return;                                                      \
  }
#define TI_CAPI_INVALID_INTEROP_ARCH_RV(x, arch)                 \
  if (x != taichi::Arch::arch) {                                 \
    ti_set_last_error(TI_ERROR_INVALID_INTEROP, "arch!=" #arch); \
    return TI_NULL_HANDLE;                                       \
  }

class Runtime;
class Context;
class AotModule;

class Runtime {
 protected:
  // 32 is a magic number in `taichi/inc/constants.h`.
  std::array<uint64_t, 32> host_result_buffer_;

  Runtime(taichi::Arch arch);

 public:
  const taichi::Arch arch;
  taichi::lang::RuntimeContext runtime_context_;

  virtual ~Runtime();

  virtual taichi::lang::Device &get() = 0;

  virtual TiAotModule load_aot_module(const char *module_path) = 0;
  virtual TiMemory allocate_memory(
      const taichi::lang::Device::AllocParams &params);
  virtual void free_memory(TiMemory devmem);

  virtual TiTexture allocate_texture(const taichi::lang::ImageParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void free_texture(TiTexture texture) {
    TI_NOT_IMPLEMENTED
  }

  virtual void buffer_copy(const taichi::lang::DevicePtr &dst,
                           const taichi::lang::DevicePtr &src,
                           size_t size) = 0;
  virtual void copy_image(const taichi::lang::DeviceAllocation &dst,
                          const taichi::lang::DeviceAllocation &src,
                          const taichi::lang::ImageCopyParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void transition_image(const taichi::lang::DeviceAllocation &image,
                                taichi::lang::ImageLayout layout) {
    TI_NOT_IMPLEMENTED
  }
  virtual void signal_event(taichi::lang::DeviceEvent *event) {
    TI_NOT_IMPLEMENTED
  }
  virtual void reset_event(taichi::lang::DeviceEvent *event) {
    TI_NOT_IMPLEMENTED
  }
  virtual void wait_event(taichi::lang::DeviceEvent *event) {
    TI_NOT_IMPLEMENTED
  }
  virtual void submit() = 0;
  virtual void wait() = 0;

  class VulkanRuntime *as_vk();
};

class AotModule {
  Runtime *runtime_;
  std::unique_ptr<taichi::lang::aot::Module> aot_module_;
  std::unordered_map<std::string,
                     std::unique_ptr<taichi::lang::aot::CompiledGraph>>
      loaded_cgraphs_;

 public:
  AotModule(Runtime &runtime,
            std::unique_ptr<taichi::lang::aot::Module> aot_module);

  taichi::lang::aot::Kernel *get_kernel(const std::string &name);
  taichi::lang::aot::CompiledGraph *get_cgraph(const std::string &name);
  taichi::lang::aot::Module &get();
  Runtime &runtime();
};

class Event {
  Runtime *runtime_;
  std::unique_ptr<taichi::lang::DeviceEvent> event_;

 public:
  Event(Runtime &runtime, std::unique_ptr<taichi::lang::DeviceEvent> event);

  taichi::lang::DeviceEvent &get();
  Runtime &runtime();
};

namespace {

template <typename THandle>
struct devalloc_cast_t {
  static inline taichi::lang::DeviceAllocation handle2devalloc(Runtime &runtime,
                                                               THandle handle) {
    return taichi::lang::DeviceAllocation{
        &runtime.get(), (taichi::lang::DeviceAllocationId)((size_t)handle - 1)};
  }
  static inline THandle devalloc2handle(
      Runtime &runtime,
      taichi::lang::DeviceAllocation devalloc) {
    return (THandle)((size_t)devalloc.alloc_id + 1);
  }
};

[[maybe_unused]] taichi::lang::DeviceAllocation devmem2devalloc(
    Runtime &runtime,
    TiMemory devmem) {
  return devalloc_cast_t<TiMemory>::handle2devalloc(runtime, devmem);
}

[[maybe_unused]] TiMemory devalloc2devmem(
    Runtime &runtime,
    const taichi::lang::DeviceAllocation &devalloc) {
  return devalloc_cast_t<TiMemory>::devalloc2handle(runtime, devalloc);
}

[[maybe_unused]] taichi::lang::DeviceAllocation devtex2devalloc(
    Runtime &runtime,
    TiTexture devtex) {
  return devalloc_cast_t<TiTexture>::handle2devalloc(runtime, devtex);
}

[[maybe_unused]] TiTexture devalloc2devtex(
    Runtime &runtime,
    const taichi::lang::DeviceAllocation &devalloc) {
  return devalloc_cast_t<TiTexture>::devalloc2handle(runtime, devalloc);
}

}  // namespace
