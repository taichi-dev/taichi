#pragma once
#include "taichi/lang_util.h"

#include "taichi/program/compile_config.h"

namespace taichi {
namespace lang {

// For backend dependent code (e.g. codegen)
// Or the backend runtime itself
// Capabilities are per-device
enum class DeviceCapability : uint32_t {
  vk_api_version,
  vk_spirv_version,
  vk_has_physical_features2,
  vk_has_int8,
  vk_has_int16,
  vk_has_int64,
  vk_has_float16,
  vk_has_float64,
  vk_has_external_memory,
  vk_has_atomic_i64,
  vk_has_atomic_float,  // load, store, exchange
  vk_has_atomic_float_add,
  vk_has_atomic_float_minmax,
  vk_has_atomic_float64,  // load, store, exchange
  vk_has_atomic_float64_add,
  vk_has_atomic_float64_minmax,
  vk_has_surface,
  vk_has_presentation,
  vk_has_spv_variable_ptr,
};

class Device;
struct DeviceAllocation;
struct DevicePtr;

// TODO: Figure out how to support images. Temporary solutions is to have all
// opque types such as images work as an allocation
struct DeviceAllocation {
  Device *device{nullptr};
  uint32_t alloc_id{0};

  DevicePtr get_ptr(uint64_t offset) const;
};

struct DeviceAllocationGuard : public DeviceAllocation {
  DeviceAllocationGuard(DeviceAllocation alloc) : DeviceAllocation(alloc) {
  }
  DeviceAllocationGuard(const DeviceAllocationGuard &) = delete;
  ~DeviceAllocationGuard();
};

struct DevicePtr : public DeviceAllocation {
  uint64_t offset{0};
};

constexpr DeviceAllocation kDeviceNullAllocation{};
constexpr DevicePtr kDeviceNullPtr{};

// TODO: Implement this
class ResourceBinder {
 public:
  virtual ~ResourceBinder() {
  }

  // In Vulkan this is called Storage Buffer (shader can store)
  virtual void rw_buffer(uint32_t set,
                         uint32_t binding,
                         DevicePtr ptr,
                         size_t size) = 0;
  virtual void rw_buffer(uint32_t set,
                         uint32_t binding,
                         DeviceAllocation alloc) = 0;

  // In Vulkan this is called Uniform Buffer (shader can only load)
  virtual void buffer(uint32_t set,
                      uint32_t binding,
                      DevicePtr ptr,
                      size_t size) = 0;
  virtual void buffer(uint32_t set,
                      uint32_t binding,
                      DeviceAllocation alloc) = 0;

  // Set vertex buffer (not implemented in compute only device)
  virtual void vertex_buffer(DevicePtr ptr, uint32_t binding = 0) {
    TI_NOT_IMPLEMENTED
  }

  // Set index buffer (not implemented in compute only device)
  // index_width = 4 -> uint32 index
  // index_width = 2 -> uint16 index
  virtual void index_buffer(DevicePtr ptr, size_t index_width) {
    TI_NOT_IMPLEMENTED
  }

  // Set frame buffer (not implemented in compute only device)
  virtual void framebuffer_color(DeviceAllocation image, uint32_t binding) {
    TI_NOT_IMPLEMENTED
  }

  // Set frame buffer (not implemented in compute only device)
  virtual void framebuffer_depth_stencil(DeviceAllocation image) {
    TI_NOT_IMPLEMENTED
  }
};

enum class PipelineSourceType {
  spirv_binary,
  spirv_src,
  glsl_src,
  hlsl_src,
  dxil_binary,
  llvm_ir_src,
  llvm_ir_binary,
  // TODO: other platforms?
};

enum class PipelineStageType {
  compute,
  fragment,
  vertex,
  tesselation_control,
  tesselation_eval,
  geometry,
  raytracing
};

// TODO: Implement this
class Pipeline {
 public:
  virtual ~Pipeline() {
  }

  virtual ResourceBinder *resource_binder() = 0;
};

// TODO: Implement this
class CommandList {
 public:
  virtual ~CommandList() {
  }

  virtual void bind_pipeline(Pipeline *p) = 0;
  virtual void bind_resources(ResourceBinder *binder) = 0;
  virtual void buffer_barrier(DevicePtr ptr, size_t size) = 0;
  virtual void buffer_barrier(DeviceAllocation alloc) = 0;
  virtual void memory_barrier() = 0;
  virtual void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) = 0;
  virtual void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) = 0;
  virtual void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) = 0;

  // These are not implemented in compute only device
  virtual void draw(uint32_t num_verticies, uint32_t start_vertex = 0) {
    TI_NOT_IMPLEMENTED
  }
  virtual void draw_indexed(uint32_t num_indicies,
                            uint32_t start_vertex = 0,
                            uint32_t start_index = 0) {
    TI_NOT_IMPLEMENTED
  }
};

struct PipelineSourceDesc {
  PipelineSourceType type;
  void *data{nullptr};
  size_t size{0};
  PipelineStageType stage{PipelineStageType::compute};
};

class Device {
 public:
  virtual ~Device(){};

  virtual uint32_t get_cap(DeviceCapability capability_id) const {
    if (caps_.find(capability_id) == caps_.end())
      return 0;
    return caps_.at(capability_id);
  }

  virtual void set_cap(DeviceCapability capability_id, uint32_t val) {
    caps_[capability_id] = val;
  }

  struct AllocParams {
    uint64_t size{0};
    bool host_write{false};
    bool host_read{false};
  };

  virtual DeviceAllocation allocate_memory(const AllocParams &params) = 0;
  virtual void dealloc_memory(DeviceAllocation allocation) = 0;

  virtual std::unique_ptr<Pipeline> create_pipeline(
      PipelineSourceDesc &src,
      std::string name = "Pipeline") = 0;

  std::unique_ptr<DeviceAllocationGuard> allocate_memory_unique(
      const AllocParams &params) {
    return std::make_unique<DeviceAllocationGuard>(
        this->allocate_memory(params));
  }

  // Mapping can fail and will return nullptr
  virtual void *map_range(DevicePtr ptr, uint64_t size) = 0;
  virtual void *map(DeviceAllocation alloc) = 0;

  virtual void unmap(DevicePtr ptr) = 0;
  virtual void unmap(DeviceAllocation alloc) = 0;

  // Directly share memory in the form of alias
  static DeviceAllocation share_to(DeviceAllocation *alloc, Device *target);

  // Strictly intra device copy (synced)
  virtual void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) = 0;

  // Copy memory inter or intra devices (synced)
  static void memcpy(DevicePtr dst, DevicePtr src, uint64_t size);

  // TODO: Add a flag to select graphics / compute pool
  virtual std::unique_ptr<CommandList> new_command_list() = 0;
  virtual void dealloc_command_list(CommandList *cmdlist) = 0;
  virtual void submit(CommandList *cmdlist) = 0;
  virtual void submit_synced(CommandList *cmdlist) = 0;

  virtual void command_sync() = 0;

 private:
  std::unordered_map<DeviceCapability, uint32_t> caps_;
};

class Surface {
 public:
  virtual ~Surface() {
  }

  virtual DeviceAllocation get_target_image() = 0;
  virtual void present_image() = 0;
};

class GraphicsDevice : public Device {
 public:
  virtual std::unique_ptr<Pipeline> create_raster_pipeline(
      std::vector<PipelineSourceDesc> &src,
      std::string name = "Pipeline") = 0;

  virtual std::unique_ptr<Surface> create_surface(uint32_t width,
                                                  uint32_t height) = 0;
};

}  // namespace lang
}  // namespace taichi
