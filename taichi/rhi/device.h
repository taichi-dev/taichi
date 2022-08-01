#pragma once
#include "taichi/util/lang_util.h"

#include "taichi/jit/jit_module.h"
#include "taichi/program/compile_config.h"
#include <string>
#include <vector>

namespace taichi {
namespace lang {

constexpr size_t kBufferSizeEntireSize = size_t(-1);

// For backend dependent code (e.g. codegen)
// Or the backend runtime itself
// Capabilities are per-device
enum class DeviceCapability : uint32_t {
  // Vulkan Caps
  vk_api_version,
  vk_has_physical_features2,
  vk_has_external_memory,
  vk_has_surface,
  vk_has_presentation,
  // SPIR-V Caps
  spirv_version,
  spirv_has_int8,
  spirv_has_int16,
  spirv_has_int64,
  spirv_has_float16,
  spirv_has_float64,
  spirv_has_atomic_i64,
  spirv_has_atomic_float16,  // load, store, exchange
  spirv_has_atomic_float16_add,
  spirv_has_atomic_float16_minmax,
  spirv_has_atomic_float,  // load, store, exchange
  spirv_has_atomic_float_add,
  spirv_has_atomic_float_minmax,
  spirv_has_atomic_float64,  // load, store, exchange
  spirv_has_atomic_float64_add,
  spirv_has_atomic_float64_minmax,
  spirv_has_variable_ptr,
  spirv_has_physical_storage_buffer,
  spirv_has_subgroup_basic,
  spirv_has_subgroup_vote,
  spirv_has_subgroup_arithmetic,
  spirv_has_subgroup_ballot,
  // Graphics Caps,
  wide_lines
};

enum class BlendOp : uint32_t { add, subtract, reverse_subtract, min, max };

enum class BlendFactor : uint32_t {
  zero,
  one,
  src_color,
  one_minus_src_color,
  dst_color,
  one_minus_dst_color,
  src_alpha,
  one_minus_src_alpha,
  dst_alpha,
  one_minus_dst_alpha
};

class Device;
struct DeviceAllocation;
struct DevicePtr;
struct LLVMRuntime;

// TODO: Figure out how to support images. Temporary solutions is to have all
// opque types such as images work as an allocation
using DeviceAllocationId = uint32_t;

struct TI_DLL_EXPORT DeviceAllocation {
  Device *device{nullptr};
  DeviceAllocationId alloc_id{0};
  // TODO: Shall we include size here?

  DevicePtr get_ptr(uint64_t offset = 0) const;

  bool operator==(const DeviceAllocation &other) const {
    return other.device == device && other.alloc_id == alloc_id;
  }

  bool operator!=(const DeviceAllocation &other) const {
    return !(*this == other);
  }
};

struct TI_DLL_EXPORT DeviceAllocationGuard : public DeviceAllocation {
  DeviceAllocationGuard(DeviceAllocation alloc) : DeviceAllocation(alloc) {
  }
  DeviceAllocationGuard(const DeviceAllocationGuard &) = delete;
  ~DeviceAllocationGuard();
};

struct TI_DLL_EXPORT DevicePtr : public DeviceAllocation {
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
struct ImageSamplerConfig {};

class ResourceBinder {
 public:
  virtual ~ResourceBinder() {
  }

  struct Bindings {};

  virtual std::unique_ptr<Bindings> materialize() = 0;

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

  virtual void image(uint32_t set,
                     uint32_t binding,
                     DeviceAllocation alloc,
                     ImageSamplerConfig sampler_config) {
    TI_NOT_IMPLEMENTED
  }

  virtual void rw_image(uint32_t set,
                        uint32_t binding,
                        DeviceAllocation alloc,
                        int lod) {
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
};

enum class PipelineSourceType {
  spirv_binary,
  spirv_src,
  glsl_src,
  hlsl_src,
  dxil_binary,
  llvm_ir_src,
  llvm_ir_binary,
  metal_src,
  metal_ir
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

// FIXME: Drop the plural form?
enum class TopologyType : int { Triangles = 0, Lines = 1, Points = 2 };

enum class PolygonMode : int {
  Fill = 0,
  Line = 1,
  Point = 2,
};

enum class TI_DLL_EXPORT BufferFormat : uint32_t {
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
  depth32f,
  unknown
};

class Pipeline {
 public:
  virtual ~Pipeline() {
  }

  virtual ResourceBinder *resource_binder() = 0;
};

enum class TI_DLL_EXPORT ImageDimension { d1D, d2D, d3D };

enum class TI_DLL_EXPORT ImageLayout {
  undefined,
  shader_read,
  shader_write,
  shader_read_write,
  color_attachment,
  color_attachment_read,
  depth_attachment,
  depth_attachment_read,
  transfer_dst,
  transfer_src,
  present_src
};

struct BufferImageCopyParams {
  uint32_t buffer_row_length{0};
  uint32_t buffer_image_height{0};
  uint32_t image_mip_level{0};
  struct {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t z{0};
  } image_offset;
  struct {
    uint32_t x{1};
    uint32_t y{1};
    uint32_t z{1};
  } image_extent;
  uint32_t image_base_layer{0};
  uint32_t image_layer_count{1};
  uint32_t image_aspect_flag{1};
};

struct ImageCopyParams {
  uint32_t width{1};
  uint32_t height{1};
  uint32_t depth{1};
};

class DeviceEvent {
 public:
  virtual ~DeviceEvent() {
  }
};

class CommandList {
 public:
  virtual ~CommandList() {
  }

  virtual void bind_pipeline(Pipeline *p) = 0;
  virtual void bind_resources(ResourceBinder *binder) = 0;
  virtual void bind_resources(ResourceBinder *binder,
                              ResourceBinder::Bindings *bindings) = 0;
  virtual void buffer_barrier(DevicePtr ptr, size_t size) = 0;
  virtual void buffer_barrier(DeviceAllocation alloc) = 0;
  virtual void memory_barrier() = 0;
  virtual void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) = 0;
  virtual void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) = 0;
  virtual void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) = 0;

  struct ComputeSize {
    uint32_t x{0};
    uint32_t y{0};
    uint32_t z{0};
  };
  // Some GPU APIs can set the block (workgroup, threadsgroup) size at
  // dispatch time.
  virtual void dispatch(ComputeSize grid_size, ComputeSize block_size) {
    dispatch(grid_size.x, grid_size.y, grid_size.z);
  }

  // These are not implemented in compute only device
  virtual void begin_renderpass(int x0,
                                int y0,
                                int x1,
                                int y1,
                                uint32_t num_color_attachments,
                                DeviceAllocation *color_attachments,
                                bool *color_clear,
                                std::vector<float> *clear_colors,
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
  virtual void draw_instance(uint32_t num_verticies,
                             uint32_t num_instances,
                             uint32_t start_vertex = 0,
                             uint32_t start_instance = 0) {
    TI_NOT_IMPLEMENTED
  }
  virtual void clear_color(float r, float g, float b, float a) {
    TI_NOT_IMPLEMENTED
  }
  virtual void set_line_width(float width) {
    TI_NOT_IMPLEMENTED
  }
  virtual void draw_indexed(uint32_t num_indicies,
                            uint32_t start_vertex = 0,
                            uint32_t start_index = 0) {
    TI_NOT_IMPLEMENTED
  }
  virtual void draw_indexed_instance(uint32_t num_indicies,
                                     uint32_t num_instances,
                                     uint32_t start_vertex = 0,
                                     uint32_t start_index = 0,
                                     uint32_t start_instance = 0) {
    TI_NOT_IMPLEMENTED
  }
  virtual void image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout) {
    TI_NOT_IMPLEMENTED
  }
  virtual void buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void copy_image(DeviceAllocation dst_img,
                          DeviceAllocation src_img,
                          ImageLayout dst_img_layout,
                          ImageLayout src_img_layout,
                          const ImageCopyParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void blit_image(DeviceAllocation dst_img,
                          DeviceAllocation src_img,
                          ImageLayout dst_img_layout,
                          ImageLayout src_img_layout,
                          const ImageCopyParams &params) {
    TI_NOT_IMPLEMENTED
  }
  virtual void signal_event(DeviceEvent *event) {
    TI_NOT_IMPLEMENTED
  }
  virtual void reset_event(DeviceEvent *event) {
    TI_NOT_IMPLEMENTED
  }
  virtual void wait_event(DeviceEvent *event) {
    TI_NOT_IMPLEMENTED
  }
};

struct PipelineSourceDesc {
  PipelineSourceType type;
  void *data{nullptr};
  size_t size{0};
  PipelineStageType stage{PipelineStageType::compute};
};

// FIXME: this probably isn't backend-neutral enough
enum class AllocUsage : int {
  None = 0,
  Storage = 1,
  Uniform = 2,
  Vertex = 4,
  Index = 8,
};
inline AllocUsage operator|(AllocUsage a, AllocUsage b) {
  return static_cast<AllocUsage>(static_cast<int>(a) | static_cast<int>(b));
}
inline bool operator&(AllocUsage a, AllocUsage b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

class StreamSemaphoreObject {
 public:
  virtual ~StreamSemaphoreObject() {
  }
};

using StreamSemaphore = std::shared_ptr<StreamSemaphoreObject>;

class Stream {
 public:
  virtual ~Stream() {
  }

  virtual std::unique_ptr<CommandList> new_command_list() = 0;
  virtual StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) = 0;
  virtual StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) = 0;

  virtual void command_sync() = 0;

  virtual double device_time_elapsed_us() const {
    TI_NOT_IMPLEMENTED
  }
};

class Device {
 public:
  virtual ~Device(){};

  uint32_t get_cap(DeviceCapability capability_id) const {
    if (caps_.find(capability_id) == caps_.end())
      return 0;
    return caps_.at(capability_id);
  }

  void set_cap(DeviceCapability capability_id, uint32_t val) {
    caps_[capability_id] = val;
  }

  void print_all_cap() const;

  struct AllocParams {
    uint64_t size{0};
    bool host_write{false};
    bool host_read{false};
    bool export_sharing{false};
    AllocUsage usage{AllocUsage::Storage};
  };

  virtual DeviceAllocation allocate_memory(const AllocParams &params) = 0;

  virtual void dealloc_memory(DeviceAllocation handle) = 0;

  virtual uint64_t get_memory_physical_pointer(DeviceAllocation handle) {
    TI_NOT_IMPLEMENTED
  }

  virtual std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") = 0;

  virtual std::unique_ptr<DeviceEvent> create_event(){TI_NOT_IMPLEMENTED}

  std::unique_ptr<DeviceAllocationGuard> allocate_memory_unique(
      const AllocParams &params) {
    return std::make_unique<DeviceAllocationGuard>(
        this->allocate_memory(params));
  }

  virtual uint64 fetch_result_uint64(int i, uint64 *result_buffer) {
    TI_NOT_IMPLEMENTED
  }

  // Each thraed will acquire its own stream
  virtual Stream *get_compute_stream() = 0;

  // Wait for all tasks to complete (task from all streams)
  virtual void wait_idle() = 0;

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
  enum class MemcpyCapability { Direct, RequiresStagingBuffer, RequiresHost };

  static MemcpyCapability check_memcpy_capability(DevicePtr dst,
                                                  DevicePtr src,
                                                  uint64_t size);

  static void memcpy_direct(DevicePtr dst, DevicePtr src, uint64_t size);

  static void memcpy_via_staging(DevicePtr dst,
                                 DevicePtr staging,
                                 DevicePtr src,
                                 uint64_t size);

  static void memcpy_via_host(DevicePtr dst,
                              void *host_buffer,
                              DevicePtr src,
                              uint64_t size);

 private:
  std::unordered_map<DeviceCapability, uint32_t> caps_;
};

class Surface {
 public:
  virtual ~Surface() {
  }

  virtual StreamSemaphore acquire_next_image() = 0;
  virtual DeviceAllocation get_target_image() = 0;
  virtual void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) = 0;
  virtual std::pair<uint32_t, uint32_t> get_size() = 0;
  virtual int get_image_count() = 0;
  virtual BufferFormat image_format() = 0;
  virtual void resize(uint32_t width, uint32_t height) = 0;
  virtual DeviceAllocation get_depth_data(DeviceAllocation &depth_alloc) = 0;
  virtual DeviceAllocation get_image_data() {
    TI_NOT_IMPLEMENTED
  }
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

struct SurfaceConfig {
  // VSync:
  // - true: will attempt to wait for V-Blank
  // - when adaptive is true: when supported, if a V-Blank is missed, instead of
  //   waiting, a tearing may appear, reduces overall latency
  bool vsync{false};
  bool adaptive{true};
  void *window_handle{nullptr};
  uint32_t width{1};
  uint32_t height{1};
};

struct TI_DLL_EXPORT ImageParams {
  ImageDimension dimension;
  BufferFormat format;
  ImageLayout initial_layout;
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
  bool export_sharing{false};
};

struct BlendFunc {
  BlendOp op{BlendOp::add};
  BlendFactor src_factor{BlendFactor::src_alpha};
  BlendFactor dst_factor{BlendFactor::one_minus_src_alpha};
};

struct BlendingParams {
  bool enable{true};
  BlendFunc color;
  BlendFunc alpha;
};

struct RasterParams {
  TopologyType prim_topology{TopologyType::Triangles};
  PolygonMode polygon_mode{PolygonMode::Fill};
  bool front_face_cull{false};
  bool back_face_cull{false};
  bool depth_test{false};
  bool depth_write{false};
  std::vector<BlendingParams> blending{};
};

class TI_DLL_EXPORT GraphicsDevice : public Device {
 public:
  virtual std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") = 0;

  virtual Stream *get_graphics_stream() = 0;

  virtual std::unique_ptr<Surface> create_surface(
      const SurfaceConfig &config) = 0;
  virtual DeviceAllocation create_image(const ImageParams &params) = 0;
  virtual void destroy_image(DeviceAllocation handle) = 0;

  virtual void image_transition(DeviceAllocation img,
                                ImageLayout old_layout,
                                ImageLayout new_layout);
  virtual void buffer_to_image(DeviceAllocation dst_img,
                               DevicePtr src_buf,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params);
  virtual void image_to_buffer(DevicePtr dst_buf,
                               DeviceAllocation src_img,
                               ImageLayout img_layout,
                               const BufferImageCopyParams &params);
};

}  // namespace lang
}  // namespace taichi
