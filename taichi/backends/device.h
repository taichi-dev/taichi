#pragma once
#include "taichi/lang_util.h"

#include "taichi/program/compile_config.h"
#include <string>
#include <vector>

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

  bool operator==(const DeviceAllocation &other) const {
    return other.device == device && other.alloc_id == alloc_id;
  }

  bool operator!=(const DeviceAllocation &other) const {
    return !(*this == other);
  }
};

struct DeviceAllocationGuard : public DeviceAllocation {
  DeviceAllocationGuard(DeviceAllocation alloc) : DeviceAllocation(alloc) {
  }
  DeviceAllocationGuard(const DeviceAllocationGuard &) = delete;
  ~DeviceAllocationGuard();
};

struct DevicePtr : public DeviceAllocation {
  uint64_t offset{0};

  bool operator==(const DevicePtr &other) const {
    return other.device == device && other.alloc_id == alloc_id &&
           other.offset == offset;
  }

  bool operator!=(const DevicePtr &other) const {
    return !(*this == other);
  }
};

constexpr DeviceAllocation kDeviceNullAllocation{};
constexpr DevicePtr kDeviceNullPtr{};


// TODO: fill this with the required options
struct ImageSamplerConfig{

};


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

  virtual void image(uint32_t set,uint32_t binding,DeviceAllocation alloc,ImageSamplerConfig sampler_config) {
    TI_NOT_IMPLEMENTED
  }

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

enum class BufferFormat : uint32_t {
  r8,
  rg8,
  rgba8,
  rgba8srgb,
  bgra8,
  bgra8srgb,
  r8u,
  rg8u,
  rgba8u,
  r8i,
  rg8i,
  rgba8i,
  r16,
  rg16,
  rgb16,
  rgba16,
  r16u,
  rg16u,
  rgb16u,
  rgba16u,
  r16i,
  rg16i,
  rgb16i,
  rgba16i,
  r16f,
  rg16f,
  rgb16f,
  rgba16f,
  r32u,
  rg32u,
  rgb32u,
  rgba32u,
  r32i,
  rg32i,
  rgb32i,
  rgba32i,
  r32f,
  rg32f,
  rgb32f,
  rgba32f,
  depth16,
  depth24stencil8,
  depth32f
};

// TODO: Implement this
class Pipeline {
 public:
  virtual ~Pipeline() {
  }

  virtual ResourceBinder *resource_binder() = 0;
};


enum class CommandListType{
  Graphics,
  Compute
};

struct CommandListConfig{
  CommandListType type;
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
  virtual void begin_renderpass(int x0,
                                int y0,
                                int x1,
                                int y1,
                                uint32_t num_color_attachments,
                                DeviceAllocation *color_attachments,
                                bool *color_clear,
                                std::vector<float>* clear_colors,
                                DeviceAllocation *depth_attachment,
                                bool depth_clear) {
    TI_NOT_IMPLEMENTED
  }
  virtual void end_renderpass() {
    TI_NOT_IMPLEMENTED
  }
  virtual void draw(uint32_t num_verticies, uint32_t start_vertex = 0) {
    TI_NOT_IMPLEMENTED
  }
  virtual void clear_color(float r,float g, float b, float a){
    TI_NOT_IMPLEMENTED
  }
  virtual void set_line_width(float width){
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

//FIXME: this probably isn't backend-neutral enough
enum class AllocUsage:int{
    Storage = 1,
    Uniform = 2,
    Vertex = 4,
    Index = 8,
};
inline AllocUsage operator|(AllocUsage a, AllocUsage b)
{
    return static_cast<AllocUsage>(static_cast<int>(a) | static_cast<int>(b));
}
inline bool operator&(AllocUsage a, AllocUsage b)
{
    return static_cast<int>(a) & static_cast<int>(b);
}

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
    bool export_sharing{false};
    AllocUsage usage{AllocUsage::Storage};
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
  virtual std::unique_ptr<CommandList> new_command_list(CommandListConfig config) = 0;
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
  virtual std::pair<uint32_t, uint32_t> get_size() = 0;
  virtual BufferFormat image_format() = 0;
};

struct VertexInputBinding {
  uint32_t binding{0};
  size_t stride{0};
  bool instance{false};
};

struct VertexInputAttribute {
  uint32_t location{0};
  uint32_t binding{0};
  BufferFormat format;
  uint32_t offset{0};
};

struct SurfaceConfig{
  uint32_t width;
  uint32_t height;
  bool vsync;
  void* window_handle;
};

class GraphicsDevice : public Device {
 public:
  virtual std::unique_ptr<Pipeline> create_raster_pipeline(
      std::vector<PipelineSourceDesc> &src,
      std::vector<BufferFormat> &render_target_formats,
      std::vector<VertexInputBinding> &vertex_inputs,
      std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") = 0;

  virtual std::unique_ptr<Surface> create_surface(const SurfaceConfig& config) = 0;
};

}  // namespace lang
}  // namespace taichi
