#pragma once
#include <vector>
#include <memory>
#include <string>
#include <exception>
#include <stdexcept>

// Taichi runtime is not necessarily using the same 3rd-party headers as the
// user codes. For C-API implementations we only use the internal headers.
#ifdef TI_WITH_VULKAN
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif  // VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"
#define TI_NO_VULKAN_INCLUDES 1
#endif  // TI_WITH_VULKAN

#ifdef TI_WITH_OPENGL
#include "glad/gl.h"
#define TI_NO_OPENGL_INCLUDES 1
#endif  // TI_WITH_OPENGL

// Then Include all C-API symbols.
#include "taichi/taichi.h"

// Include for the base types.
#include "taichi/rhi/arch.h"
#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST
#include "taichi/rhi/device.h"
#include "taichi/aot/graph_data.h"
#include "taichi/aot/module_loader.h"
#include "taichi/common/virtual_dir.h"

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

#define TI_CAPI_INVALID_ARGUMENT(pred)                   \
  if (pred) {                                            \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, #pred); \
    return;                                              \
  }
#define TI_CAPI_INVALID_ARGUMENT_RV(pred)                \
  if (pred) {                                            \
    ti_set_last_error(TI_ERROR_INVALID_ARGUMENT, #pred); \
    return TI_NULL_HANDLE;                               \
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

#define TI_CAPI_TRY_CATCH_BEGIN() try {
#define TI_CAPI_TRY_CATCH_END()                                 \
  }                                                             \
  catch (const std::exception &e) {                             \
    ti_set_last_error(TI_ERROR_INVALID_STATE, e.what());        \
  }                                                             \
  catch (const std::string &e) {                                \
    ti_set_last_error(TI_ERROR_INVALID_STATE, e.c_str());       \
  }                                                             \
  catch (...) {                                                 \
    ti_set_last_error(TI_ERROR_INVALID_STATE, "c++ exception"); \
  }

struct Error {
  TiError error;
  std::string message;

  Error(TiError error, const std::string &message)
      : error(error), message(message) {
  }
  Error() : error(TI_ERROR_SUCCESS), message() {
  }
  Error(const Error &) = delete;
  Error(Error &&) = default;
  Error &operator=(const Error &) = delete;
  Error &operator=(Error &&) = default;

  // Set this error as the last error if it's not `TI_ERROR_SUCCESS`.
  inline void set_last_error() const {
    if (error != TI_ERROR_SUCCESS) {
      ti_set_last_error(error, message.c_str());
    }
  }
};

namespace capi {
class MetalRuntime;
}  // namespace capi

class Runtime {
 protected:
  // 32 is a magic number in `taichi/inc/constants.h`.
  std::array<uint64_t, 32> host_result_buffer_;

  explicit Runtime(taichi::Arch arch);

 public:
  const taichi::Arch arch;

  virtual ~Runtime();

  virtual taichi::lang::Device &get() = 0;

  [[deprecated("create_aot_module")]] virtual TiAotModule load_aot_module(
      const char *module_path) {
    auto dir = taichi::io::VirtualDir::open(module_path);
    TiAotModule aot_module = TI_NULL_HANDLE;
    Error err = create_aot_module(dir.get(), aot_module);
    err.set_last_error();
    return aot_module;
  }

  virtual Error create_aot_module(const taichi::io::VirtualDir *dir,
                                  TiAotModule &out) {
    TI_NOT_IMPLEMENTED
  }
  virtual TiMemory allocate_memory(
      const taichi::lang::Device::AllocParams &params);
  virtual void free_memory(TiMemory devmem);

  virtual TiImage allocate_image(const taichi::lang::ImageParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void free_image(TiImage image) {
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
  virtual void track_image(const taichi::lang::DeviceAllocation &image,
                           taichi::lang::ImageLayout layout) {
    TI_NOT_IMPLEMENTED
  }
  virtual void untrack_image(const taichi::lang::DeviceAllocation &image) {
    TI_NOT_IMPLEMENTED
  }
  virtual void transition_image(const taichi::lang::DeviceAllocation &image,
                                taichi::lang::ImageLayout layout) {
    TI_NOT_IMPLEMENTED
  }
  virtual void flush() = 0;
  virtual void wait() = 0;

  class VulkanRuntime *as_vk();
  class capi::MetalRuntime *as_mtl();
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

[[maybe_unused]] taichi::lang::DeviceAllocation devimg2devalloc(
    Runtime &runtime,
    TiImage devimg) {
  return devalloc_cast_t<TiImage>::handle2devalloc(runtime, devimg);
}

[[maybe_unused]] TiImage devalloc2devimg(
    Runtime &runtime,
    const taichi::lang::DeviceAllocation &devalloc) {
  return devalloc_cast_t<TiImage>::devalloc2handle(runtime, devalloc);
}

}  // namespace
