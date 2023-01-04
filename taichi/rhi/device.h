#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include <initializer_list>

#include "taichi/common/logging.h"

#include "taichi/rhi/device_capability.h"
#include "taichi/rhi/arch.h"

namespace taichi::lang {

enum class RhiResult {
  success = 0,
  error = -1,
  invalid_usage = -2,
  not_supported = -3,
  out_of_memory = -4,
};

constexpr size_t kBufferSizeEntireSize = size_t(-1);

#define MAKE_ENUM_FLAGS(name)                  \
  inline name operator|(name a, name b) {      \
    return static_cast<name>(int(a) | int(b)); \
  }                                            \
  inline name operator&(name a, name b) {      \
    return static_cast<name>(int(a) & int(b)); \
  }                                            \
  inline bool operator&&(name a, name b) { return (int(a) & int(b)) != 0; }

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

// TODO: Figure out how to support images. Temporary solutions is to have all
// opque types such as images work as an allocation
using DeviceAllocationId = uint64_t;

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
  explicit DeviceAllocationGuard(DeviceAllocation alloc)
      : DeviceAllocation(alloc) {
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

// A set of shader resources (that is bound at once)
class TI_DLL_EXPORT ShaderResourceSet {
 public:
  virtual ~ShaderResourceSet() = default;

  /**
   * Bind a RW subregion of a buffer resource (StorgeBuffer / SSBO)
   * @params[in] binding The binding index of the resource
   * @params[in] ptr The Device Pointer that is going to be bound
   * @params[in] size The size of the bound region of the buffer
   */
  virtual ShaderResourceSet &rw_buffer(uint32_t binding,
                                       DevicePtr ptr,
                                       size_t size) = 0;

  /**
   * Bind an entire RW buffer resource (StorgeBuffer / SSBO)
   * @params[in] binding The binding index of the resource
   * @params[in] alloc The Device Allocation that is going to be bound
   */
  virtual ShaderResourceSet &rw_buffer(uint32_t binding,
                                       DeviceAllocation alloc) = 0;

  /**
   * Bind a read-only subregion of a buffer resource (Constants / UBO)
   * @params[in] binding The binding index of the resource
   * @params[in] ptr The Device Pointer that is going to be bound
   * @params[in] size The size of the bound region of the buffer
   */
  virtual ShaderResourceSet &buffer(uint32_t binding,
                                    DevicePtr ptr,
                                    size_t size) = 0;

  /**
   * Bind an entire read-only buffer resource (Constants / UBO)
   * @params[in] binding The binding index of the resource
   * @params[in] alloc The Device Allocation that is going to be bound
   */
  virtual ShaderResourceSet &buffer(uint32_t binding,
                                    DeviceAllocation alloc) = 0;

  /**
   * Bind a read-only image resource (SRV / Texture)
   * @params[in] binding The binding index of the resource
   * @params[in] alloc The Device Allocation that is going to be bound
   * @params[in] sampler_config The texture sampling configuration
   */
  virtual ShaderResourceSet &image(uint32_t binding,
                                   DeviceAllocation alloc,
                                   ImageSamplerConfig sampler_config) {
    TI_NOT_IMPLEMENTED;
  }

  /**
   * Bind a RW image resource (UAV / Storage Image)
   * @params binding The binding index of the resource
   * @params alloc The Device Allocation that is going to be bound
   */
  virtual ShaderResourceSet &rw_image(uint32_t binding,
                                      DeviceAllocation alloc,
                                      int lod) {
    TI_NOT_IMPLEMENTED
  }
};

// A set of states / resources for rasterization
class TI_DLL_EXPORT RasterResources {
 public:
  virtual ~RasterResources() = default;

  /**
   * Set a vertex buffer for the rasterization
   * @params ptr The Device Pointer to the vertices data
   * @params binding The binding index of the vertex buffer
   */
  virtual RasterResources &vertex_buffer(DevicePtr ptr, uint32_t binding = 0) {
    TI_NOT_IMPLEMENTED
  }

  /**
   * Set an index buffer for the rasterization
   * @params ptr The Device Pointer to the vertices data
   * @params index_width The index data width (in bits).
   *                     index_width = 32 -> uint32 index
   *                     index_width = 16 -> uint16 index
   */
  virtual RasterResources &index_buffer(DevicePtr ptr, size_t index_width) {
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

enum class BufferFormat : uint32_t {
#define PER_BUFFER_FORMAT(x) x,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_BUFFER_FORMAT
};

class TI_DLL_EXPORT Pipeline {
 public:
  virtual ~Pipeline() {
  }
};

enum class ImageDimension {
#define PER_IMAGE_DIMENSION(x) x,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_DIMENSION
};

enum class ImageLayout {
#define PER_IMAGE_LAYOUT(x) x,
#include "taichi/inc/rhi_constants.inc.h"
#undef PER_IMAGE_LAYOUT
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

class TI_DLL_EXPORT CommandList {
 public:
  virtual ~CommandList() {
  }

  /**
   * Bind a pipeline to the command list.
   * Doing so resets all bound resources.
   * @params[in] pipeline The pipeline to be bound
   */
  virtual void bind_pipeline(Pipeline *p) = 0;

  /**
   * Bind a ShaderResourceSet to a set index.
   * - If the set index is already bound, the previous binding will be
   *   overwritten.
   * - A set index can only be bound with a single ShaderResourceSet.
   * - If the input set is empty, this command is a no-op.
   * @params[in] res The ShaderResourceSet to be bound.
   * @params[in] set_index The index the resources will be bound to.
   * @return The binding result code
   *         `success` If the binding succeded
   *         `invalid_usage` If `res` is incompatible with current pipeline
   *         `not_supported` If some bindings are not supported by the backend
   *         `out_of_memory` If binding failed due to OOM conditions
   *         `error` If binding failed due to other reasons
   */
  virtual RhiResult bind_shader_resources(ShaderResourceSet *res,
                                          int set_index = 0) = 0;

  /**
   * Bind RasterResources to the command list.
   * - If the input resource is empty, this command is a no-op.
   * @params res The RasterResources to be bound.
   * @return The binding result code
   *         `success` If the binding succeded
   *         `invalid_usage` If `res` is incompatible with current pipeline
   *         `not_supported` If some bindings are not supported by the backend
   *         `error` If binding failed due to other reasons
   */
  virtual RhiResult bind_raster_resources(RasterResources *res) = 0;

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
};

struct PipelineSourceDesc {
  PipelineSourceType type;
  const void *data{nullptr};
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

MAKE_ENUM_FLAGS(AllocUsage)

class TI_DLL_EXPORT StreamSemaphoreObject {
 public:
  virtual ~StreamSemaphoreObject() {
  }
};

using StreamSemaphore = std::shared_ptr<StreamSemaphoreObject>;

class TI_DLL_EXPORT Stream {
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

class TI_DLL_EXPORT Device {
  DeviceCapabilityConfig caps_{};

 public:
  virtual ~Device(){};

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
    // FIXME: (penguinliong) This method reports the actual device memory
    // address, it's used for bindless (like argument buffer on Metal). If the
    // backend doesn't have access to physical memory address, it should return
    // null and it depends on the backend implementation to use the address in
    // argument binders.
    return 0;
  }

  virtual std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") = 0;

  std::unique_ptr<DeviceAllocationGuard> allocate_memory_unique(
      const AllocParams &params) {
    return std::make_unique<DeviceAllocationGuard>(
        this->allocate_memory(params));
  }

  virtual uint64_t fetch_result_uint64(int i, uint64_t *result_buffer) {
    TI_NOT_IMPLEMENTED
  }

  // Each thraed will acquire its own stream
  virtual Stream *get_compute_stream() = 0;

  // Wait for all tasks to complete (task from all streams)
  virtual void wait_idle() = 0;

  /**
   * Create a new shader resource set
   * @return The new shader resource set pointer
   */
  virtual ShaderResourceSet *create_resource_set() = 0;

  /**
   * Create a new shader resource set (wrapped in unique ptr)
   * @return The new shader resource set unique pointer
   */
  inline std::unique_ptr<ShaderResourceSet> create_resource_set_unique() {
    return std::unique_ptr<ShaderResourceSet>(this->create_resource_set());
  }

  /**
   * Map a range within a DeviceAllocation memory into host address space.
   *
   * @param[in] ptr The Device Pointer to map.
   * @param[in] size The size of the mapped region.
   * @param[out] mapped_ptr Outputs the pointer to the mapped region.
   * @return The result status.
   *         `success` when the mapping is successful.
   *         `invalid_usage` when the memory is not host visible.
   *         `invalid_usage` when trying to map the memory multiple times.
   *         `invalid_usage` when `ptr.offset + size` is out-of-bounds.
   *         `error` when the mapping failed for other reasons.
   */
  virtual RhiResult map_range(DevicePtr ptr,
                              uint64_t size,
                              void **mapped_ptr) = 0;

  /**
   * Map an entire DeviceAllocation into host address space.
   * @param[in] ptr The Device Pointer to map.
   * @param[in] size The size of the mapped region.
   * @param[out] mapped_ptr Outputs the pointer to the mapped region.
   * @return The result status.
   *         `success` when the mapping is successful.
   *         `invalid_usage` when the memory is not host visible.
   *         `invalid_usage` when trying to map the memory multiple times.
   *         `invalid_usage` when `ptr.offset + size` is out-of-bounds.
   *         `error` when the mapping failed for other reasons.
   */
  virtual RhiResult map(DeviceAllocation alloc, void **mapped_ptr) = 0;

  /**
   * Unmap a previously mapped DevicePtr or DeviceAllocation.
   * @param[in] ptr The DevicePtr to unmap.
   */
  virtual void unmap(DevicePtr ptr) = 0;

  /**
   * Unmap a previously mapped DevicePtr or DeviceAllocation.
   * @param[in] alloc The DeviceAllocation to unmap
   */
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

  // Get all supported capabilities of the current created device.
  virtual Arch arch() const = 0;
  inline const DeviceCapabilityConfig &get_caps() const {
    return caps_;
  }
  inline void set_caps(DeviceCapabilityConfig &&caps) {
    caps_ = std::move(caps);
  }
};

class TI_DLL_EXPORT Surface {
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
  void *native_surface_handle{nullptr};
};

enum class ImageAllocUsage : int {
  None = 0,
  Storage = 1,
  Sampled = 2,
  Attachment = 4,
};
inline ImageAllocUsage operator|(ImageAllocUsage a, ImageAllocUsage b) {
  return static_cast<ImageAllocUsage>(static_cast<int>(a) |
                                      static_cast<int>(b));
}
inline bool operator&(ImageAllocUsage a, ImageAllocUsage b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

struct ImageParams {
  ImageDimension dimension;
  BufferFormat format;
  ImageLayout initial_layout{ImageLayout::undefined};
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
  bool export_sharing{false};
  ImageAllocUsage usage{ImageAllocUsage::Storage | ImageAllocUsage::Sampled |
                        ImageAllocUsage::Attachment};
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

  /**
   * Create a new raster resources set
   * @return The new RasterResources pointer
   */
  virtual RasterResources *create_raster_resources() = 0;

  /**
   * Create a new raster resources set (wrapped in unique ptr)
   * @return The new RasterResources unique pointer
   */
  inline std::unique_ptr<RasterResources> create_raster_resources_unique() {
    return std::unique_ptr<RasterResources>(this->create_raster_resources());
  }

  virtual std::unique_ptr<Surface> create_surface(
      const SurfaceConfig &config) = 0;
  // You are not expected to call this directly. If you want to use this image
  // in a taichi kernel, you usually want to create the image via
  // `GfxRuntime::create_image`. `GfxRuntime` is available in `ProgramImpl`
  // of GPU backends.
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

}  // namespace taichi::lang
