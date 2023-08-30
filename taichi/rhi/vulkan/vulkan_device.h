#pragma once

#include "taichi/rhi/device.h"
#include "taichi/rhi/vulkan/vulkan_api.h"
#include "taichi/rhi/vulkan/vulkan_utils.h"
#include "taichi/common/ref_counted_pool.h"

#include "vk_mem_alloc.h"

#include <memory>
#include <optional>
#include <list>
#include <variant>

namespace taichi::lang {
namespace vulkan {

using std::unordered_map;

class VulkanDevice;
class VulkanResourceBinder;
class VulkanStream;

struct SpirvCodeView {
  const uint32_t *data = nullptr;
  size_t size = 0;
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

  SpirvCodeView() = default;

  explicit SpirvCodeView(const std::vector<uint32_t> &code)
      : data(code.data()), size(code.size() * sizeof(uint32_t)) {
  }
};

struct VulkanRenderPassDesc {
  std::vector<std::pair<VkFormat, bool>> color_attachments;
  VkFormat depth_attachment{VK_FORMAT_UNDEFINED};
  bool clear_depth{false};

  bool operator==(const VulkanRenderPassDesc &other) const {
    if (other.depth_attachment != depth_attachment) {
      return false;
    }
    if (other.clear_depth != clear_depth) {
      return false;
    }
    return other.color_attachments == color_attachments;
  }
};

struct RenderPassDescHasher {
  std::size_t operator()(const VulkanRenderPassDesc &desc) const {
    size_t hash = std::hash<uint64_t>()((uint64_t(desc.depth_attachment) << 1) |
                                        uint64_t(desc.clear_depth));
    for (auto &pair : desc.color_attachments) {
      size_t hash_pair = std::hash<uint64_t>()((uint64_t(pair.first) << 1) |
                                               uint64_t(pair.second));
      rhi_impl::hash_combine(hash, hash_pair);
    }
    return hash;
  }
};

struct VulkanFramebufferDesc {
  std::vector<vkapi::IVkImageView> attachments{};
  uint32_t width{0};
  uint32_t height{0};
  vkapi::IVkRenderPass renderpass{nullptr};

  bool operator==(const VulkanFramebufferDesc &other) const {
    return width == other.width && height == other.height &&
           renderpass == other.renderpass && attachments == other.attachments;
  }
};

class VulkanResourceSet : public ShaderResourceSet {
 public:
  struct Buffer {
    vkapi::IVkBuffer buffer{nullptr};
    VkDeviceSize offset{0};
    VkDeviceSize size{0};

    bool operator==(const Buffer &rhs) const {
      return buffer == rhs.buffer && offset == rhs.offset && size == rhs.size;
    }

    bool operator!=(const Buffer &rhs) const {
      return !(*this == rhs);
    }
  };

  struct Image {
    vkapi::IVkImageView view{nullptr};

    bool operator==(const Image &rhs) const {
      return view == rhs.view;
    }

    bool operator!=(const Image &rhs) const {
      return view != rhs.view;
    }
  };

  struct Texture {
    vkapi::IVkImageView view{nullptr};
    vkapi::IVkSampler sampler{nullptr};

    bool operator==(const Texture &rhs) const {
      return view == rhs.view && sampler == rhs.sampler;
    }

    bool operator!=(const Texture &rhs) const {
      return !(*this == rhs);
    }
  };

  struct Binding {
    VkDescriptorType type{VK_DESCRIPTOR_TYPE_MAX_ENUM};
    std::variant<Buffer, Image, Texture> res{Buffer()};

    bool operator==(const Binding &other) const {
      return other.type == type && other.res == res;
    }

    bool operator!=(const Binding &other) const {
      return other.type != type || other.res != res;
    }

    size_t hash() const {
      size_t hash = 0;
      rhi_impl::hash_combine(hash, int(type));
      if (const Buffer *buf = std::get_if<Buffer>(&res)) {
        rhi_impl::hash_combine(hash, (void *)buf->buffer.get());
        rhi_impl::hash_combine(hash, size_t(buf->offset));
        rhi_impl::hash_combine(hash, size_t(buf->size));
      } else if (const Image *img = std::get_if<Image>(&res)) {
        rhi_impl::hash_combine(hash, (void *)img->view.get());
      } else if (const Texture *tex = std::get_if<Texture>(&res)) {
        rhi_impl::hash_combine(hash, (void *)tex->view.get());
        rhi_impl::hash_combine(hash, (void *)tex->sampler.get());
      }
      return hash;
    }
  };

  // This hashes the Set Layout
  struct SetLayoutHasher {
    std::size_t operator()(const VulkanResourceSet &set) const {
      // NOTE: Bindings in this case is ordered, we can use non-commutative
      // operations
      size_t hash = 0;
      for (const auto &pair : set.bindings_) {
        rhi_impl::hash_combine(hash, pair.first);
        // We only care about type in this case
        rhi_impl::hash_combine(hash, pair.second.type);
      }
      return hash;
    }
  };

  // This compares the layout of two sets
  struct SetLayoutCmp {
    bool operator()(const VulkanResourceSet &lhs,
                    const VulkanResourceSet &rhs) const {
      if (lhs.bindings_.size() != rhs.bindings_.size()) {
        return false;
      }
      for (auto &lhs_pair : lhs.bindings_) {
        auto rhs_binding_iter = rhs.bindings_.find(lhs_pair.first);
        if (rhs_binding_iter == rhs.bindings_.end()) {
          return false;
        }
        const Binding &rhs_binding = rhs_binding_iter->second;
        if (rhs_binding.type != lhs_pair.second.type) {
          return false;
        }
      }
      return true;
    }
  };

  // This hashes the entire set (including resources)
  struct DescSetHasher {
    std::size_t operator()(const VulkanResourceSet &set) const {
      size_t hash = 0;
      for (const auto &pair : set.bindings_) {
        rhi_impl::hash_combine(hash, pair.first);
        hash ^= pair.second.hash() + 0x9e3779b9 + (hash << 6) + (hash >> 2);
      }
      return hash;
    }
  };

  // This compares two sets (including resources)
  struct SetCmp {
    bool operator()(const VulkanResourceSet &lhs,
                    const VulkanResourceSet &rhs) const {
      return lhs.bindings_ == rhs.bindings_;
    }
  };

  explicit VulkanResourceSet(VulkanDevice *device);
  VulkanResourceSet(const VulkanResourceSet &other) = default;
  ~VulkanResourceSet() override;

  ShaderResourceSet &rw_buffer(uint32_t binding,
                               DevicePtr ptr,
                               size_t size) final;
  ShaderResourceSet &rw_buffer(uint32_t binding, DeviceAllocation alloc) final;
  ShaderResourceSet &buffer(uint32_t binding, DevicePtr ptr, size_t size) final;
  ShaderResourceSet &buffer(uint32_t binding, DeviceAllocation alloc) final;
  ShaderResourceSet &image(uint32_t binding,
                           DeviceAllocation alloc,
                           ImageSamplerConfig sampler_config) final;
  ShaderResourceSet &rw_image(uint32_t binding,
                              DeviceAllocation alloc,
                              int lod) final;

  rhi_impl::RhiReturn<vkapi::IVkDescriptorSet> finalize();

  vkapi::IVkDescriptorSetLayout get_layout() {
    return layout_;
  }

  const std::map<uint32_t, Binding> &get_bindings() const {
    return bindings_;
  }

 private:
  std::map<uint32_t, Binding> bindings_;
  VulkanDevice *device_;

  vkapi::IVkDescriptorSetLayout layout_{nullptr};
  vkapi::IVkDescriptorSet set_{nullptr};

  bool dirty_{true};
};

class VulkanRasterResources : public RasterResources {
 public:
  explicit VulkanRasterResources(VulkanDevice *device) : device_(device) {
  }

  struct BufferBinding {
    vkapi::IVkBuffer buffer{nullptr};
    size_t offset{0};
  };

  std::unordered_map<uint32_t, BufferBinding> vertex_buffers;
  BufferBinding index_binding;
  VkIndexType index_type{VK_INDEX_TYPE_MAX_ENUM};

  ~VulkanRasterResources() override = default;

  RasterResources &vertex_buffer(DevicePtr ptr, uint32_t binding = 0) final;
  RasterResources &index_buffer(DevicePtr ptr, size_t index_width) final;

 private:
  VulkanDevice *device_;
};

class VulkanPipelineCache : public PipelineCache {
 public:
  VulkanPipelineCache(VulkanDevice *device,
                      size_t initial_size,
                      const void *initial_data);
  ~VulkanPipelineCache() override;

  void *data() noexcept final;
  size_t size() const noexcept final;

  vkapi::IVkPipelineCache vk_pipeline_cache() {
    return cache_;
  }

 private:
  VulkanDevice *device_{nullptr};
  vkapi::IVkPipelineCache cache_{nullptr};
  std::vector<uint8_t> data_shadow_;
};

// VulkanPipeline maps to a vkapi::IVkPipeline, or a SPIR-V module (a GLSL
// compute shader).
class VulkanPipeline : public Pipeline {
 public:
  struct Params {
    VulkanDevice *device{nullptr};
    std::vector<SpirvCodeView> code;
    std::string name{"Pipeline"};
    vkapi::IVkPipelineCache cache{nullptr};
  };

  explicit VulkanPipeline(const Params &params);
  explicit VulkanPipeline(
      const Params &params,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);
  ~VulkanPipeline() override;

  vkapi::IVkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }

  vkapi::IVkPipeline pipeline() const {
    return pipeline_;
  }

  vkapi::IVkPipeline graphics_pipeline(
      const VulkanRenderPassDesc &renderpass_desc,
      vkapi::IVkRenderPass renderpass);

  vkapi::IVkPipeline graphics_pipeline_dynamic(
      const VulkanRenderPassDesc &renderpass_desc);

  const std::string &name() const {
    return name_;
  }

  bool is_graphics() const {
    return graphics_pipeline_template_ != nullptr;
  }

  std::unordered_map<uint32_t, VulkanResourceSet>
      &get_resource_set_templates() {
    return set_templates_;
  }

 private:
  void create_descriptor_set_layout(const Params &params);
  void create_shader_stages(const Params &params);
  void create_pipeline_layout();
  void create_compute_pipeline(const Params &params);
  void create_graphics_pipeline(
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);

  static VkShaderModule create_shader_module(VkDevice device,
                                             const SpirvCodeView &code);

  struct GraphicsPipelineTemplate {
    VkPipelineViewportStateCreateInfo viewport_state{};
    std::vector<VkVertexInputBindingDescription> input_bindings;
    std::vector<VkVertexInputAttributeDescription> input_attrs;
    VkPipelineVertexInputStateCreateInfo input{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    VkPipelineMultisampleStateCreateInfo multisampling{};
    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    VkPipelineColorBlendStateCreateInfo color_blending{};
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments{};
    std::vector<VkDynamicState> dynamic_state_enables = {
        VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    VkGraphicsPipelineCreateInfo pipeline_info{};
  };

  VulkanDevice &ti_device_;          // not owned
  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages_;

  std::unique_ptr<GraphicsPipelineTemplate> graphics_pipeline_template_;
  std::unordered_map<vkapi::IVkRenderPass, vkapi::IVkPipeline>
      graphics_pipeline_;

  // For KHR_dynamic_rendering
  std::unordered_map<VulkanRenderPassDesc,
                     vkapi::IVkPipeline,
                     RenderPassDescHasher>
      graphics_pipeline_dynamic_;

  std::unordered_map<uint32_t, VulkanResourceSet> set_templates_;
  std::vector<vkapi::IVkDescriptorSetLayout> set_layouts_;
  std::vector<VkShaderModule> shader_modules_;
  vkapi::IVkPipeline pipeline_{VK_NULL_HANDLE};
  vkapi::IVkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device,
                    VulkanStream *stream,
                    vkapi::IVkCommandBuffer buffer);
  ~VulkanCommandList() override;

  void bind_pipeline(Pipeline *p) noexcept final;
  RhiResult bind_shader_resources(ShaderResourceSet *res,
                                  int set_index = 0) noexcept final;
  RhiResult bind_raster_resources(RasterResources *res) noexcept final;
  void buffer_barrier(DevicePtr ptr, size_t size) noexcept final;
  void buffer_barrier(DeviceAllocation alloc) noexcept final;
  void memory_barrier() noexcept final;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) noexcept final;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) noexcept final;
  RhiResult dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) noexcept final;
  void begin_renderpass(int x0,
                        int y0,
                        int x1,
                        int y1,
                        uint32_t num_color_attachments,
                        DeviceAllocation *color_attachments,
                        bool *color_clear,
                        std::vector<float> *clear_colors,
                        DeviceAllocation *depth_attachment,
                        bool depth_clear) override;
  void end_renderpass() override;
  void draw(uint32_t num_verticies, uint32_t start_vertex = 0) override;
  void draw_instance(uint32_t num_verticies,
                     uint32_t num_instances,
                     uint32_t start_vertex = 0,
                     uint32_t start_instance = 0) override;
  void draw_indexed(uint32_t num_indicies,
                    uint32_t start_vertex = 0,
                    uint32_t start_index = 0) override;
  void draw_indexed_instance(uint32_t num_indicies,
                             uint32_t num_instances,
                             uint32_t start_vertex = 0,
                             uint32_t start_index = 0,
                             uint32_t start_instance = 0) override;
  void set_line_width(float width) override;
  void image_transition(DeviceAllocation img,
                        ImageLayout old_layout,
                        ImageLayout new_layout) override;
  void buffer_to_image(DeviceAllocation dst_img,
                       DevicePtr src_buf,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;
  void image_to_buffer(DevicePtr dst_buf,
                       DeviceAllocation src_img,
                       ImageLayout img_layout,
                       const BufferImageCopyParams &params) override;

  void copy_image(DeviceAllocation dst_img,
                  DeviceAllocation src_img,
                  ImageLayout dst_img_layout,
                  ImageLayout src_img_layout,
                  const ImageCopyParams &params) override;

  void blit_image(DeviceAllocation dst_img,
                  DeviceAllocation src_img,
                  ImageLayout dst_img_layout,
                  ImageLayout src_img_layout,
                  const ImageCopyParams &params) override;

  vkapi::IVkRenderPass current_renderpass();

  // Vulkan specific functions
  vkapi::IVkCommandBuffer finalize();

  vkapi::IVkCommandBuffer vk_command_buffer();

  // Profiler support
  void begin_profiler_scope(const std::string &kernel_name) override;
  void end_profiler_scope() override;

 private:
  bool finalized_{false};
  VulkanDevice *ti_device_;
  VulkanStream *stream_;
  VkDevice device_;
  vkapi::IVkCommandBuffer buffer_;
  VulkanPipeline *current_pipeline_{nullptr};

  // Renderpass & raster pipeline
  std::vector<vkapi::IVkImage> current_dynamic_targets_;
  VulkanRenderPassDesc current_renderpass_desc_;
  vkapi::IVkRenderPass current_renderpass_{VK_NULL_HANDLE};
  vkapi::IVkFramebuffer current_framebuffer_{VK_NULL_HANDLE};
  uint32_t viewport_width_{0}, viewport_height_{0};
};

class VulkanSurface : public Surface {
 public:
  VulkanSurface(VulkanDevice *device, const SurfaceConfig &config);
  ~VulkanSurface() override;

  StreamSemaphore acquire_next_image() override;
  DeviceAllocation get_target_image() override;

  void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  std::pair<uint32_t, uint32_t> get_size() override;
  int get_image_count() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;

 private:
  void create_swap_chain();
  void destroy_swap_chain();

  SurfaceConfig config_;

  VulkanDevice *device_{nullptr};
  VkSurfaceKHR surface_{VK_NULL_HANDLE};
  VkSwapchainKHR swapchain_{VK_NULL_HANDLE};
  vkapi::IVkSemaphore image_available_{nullptr};
  BufferFormat image_format_{BufferFormat::unknown};

  uint32_t image_index_{0};

  uint32_t width_{0};
  uint32_t height_{0};

  std::vector<DeviceAllocation> swapchain_images_;
};

struct DescPool {
  VkDescriptorPool pool;
  // Threads share descriptor sets
  RefCountedPool<vkapi::IVkDescriptorSet, true> sets;

  explicit DescPool(VkDescriptorPool pool) : pool(pool) {
  }
};

class VulkanStreamSemaphoreObject : public StreamSemaphoreObject {
 public:
  explicit VulkanStreamSemaphoreObject(vkapi::IVkSemaphore sema)
      : vkapi_ref(sema) {
  }
  ~VulkanStreamSemaphoreObject() override {
  }

  vkapi::IVkSemaphore vkapi_ref{nullptr};
};

class VulkanStream : public Stream {
 public:
  VulkanStream(VulkanDevice &device,
               VkQueue queue,
               uint32_t queue_family_index);
  ~VulkanStream() override;

  RhiResult new_command_list(CommandList **out_cmdlist) noexcept final;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;

  void command_sync() override;

 private:
  struct TrackedCmdbuf {
    vkapi::IVkFence fence;
    vkapi::IVkCommandBuffer buf;
  };

  VulkanDevice &device_;
  VkQueue queue_;
  uint32_t queue_family_index_;

  // Command pools are per-thread
  vkapi::IVkCommandPool command_pool_;
  std::vector<TrackedCmdbuf> submitted_cmdbuffers_;
};

struct VulkanCapabilities {
  uint32_t vk_api_version{0};
  bool physical_device_features2{false};
  bool external_memory{false};
  bool wide_line{false};
  bool surface{false};
  bool present{false};
  bool dynamic_rendering{false};
};

class TI_DLL_EXPORT VulkanDevice : public GraphicsDevice {
 public:
  struct Params {
    PFN_vkGetInstanceProcAddr get_proc_addr{nullptr};
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue compute_queue{VK_NULL_HANDLE};
    uint32_t compute_queue_family_index{0};
    VkQueue graphics_queue{VK_NULL_HANDLE};
    uint32_t graphics_queue_family_index{0};
  };

  VulkanDevice();
  void init_vulkan_structs(Params &params);
  ~VulkanDevice() override;

  Arch arch() const override {
    return Arch::vulkan;
  }

  RhiResult create_pipeline_cache(
      PipelineCache **out_cache,
      size_t initial_size = 0,
      const void *initial_data = nullptr) noexcept final;

  RhiResult create_pipeline(Pipeline **out_pipeline,
                            const PipelineSourceDesc &src,
                            std::string name,
                            PipelineCache *cache) noexcept final;

  RhiResult allocate_memory(const AllocParams &params,
                            DeviceAllocation *out_devalloc) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t get_memory_physical_pointer(DeviceAllocation handle) override;

  ShaderResourceSet *create_resource_set() final;

  RasterResources *create_raster_resources() final;

  RhiResult map_range(DevicePtr ptr, uint64_t size, void **mapped_ptr) final;
  RhiResult map(DeviceAllocation alloc, void **mapped_ptr) final;

  void unmap(DevicePtr ptr) final;
  void unmap(DeviceAllocation alloc) final;

  // Strictly intra device copy
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  Stream *get_compute_stream() override;
  Stream *get_graphics_stream() override;

  void wait_idle() override;

  std::unique_ptr<Pipeline> create_raster_pipeline(
      const std::vector<PipelineSourceDesc> &src,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") override;

  std::unique_ptr<Surface> create_surface(const SurfaceConfig &config) override;

  DeviceAllocation create_image(const ImageParams &params) override;
  void destroy_image(DeviceAllocation handle) override;

  // Vulkan specific functions
  VkInstance vk_instance() const {
    return instance_;
  }

  VkDevice vk_device() const {
    return device_;
  }

  VkPhysicalDevice vk_physical_device() const {
    return physical_device_;
  }

  uint32_t compute_queue_family_index() const {
    return compute_queue_family_index_;
  }

  uint32_t graphics_queue_family_index() const {
    return graphics_queue_family_index_;
  }

  VkQueue graphics_queue() const {
    return graphics_queue_;
  }

  VkQueue compute_queue() const {
    return compute_queue_;
  }

  std::tuple<VkDeviceMemory, size_t, size_t> get_vkmemory_offset_size(
      const DeviceAllocation &alloc) const;

  vkapi::IVkBuffer get_vkbuffer(const DeviceAllocation &alloc) const;

  size_t get_vkbuffer_size(const DeviceAllocation &alloc) const;

  std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat> get_vk_image(
      const DeviceAllocation &alloc) const;

  DeviceAllocation import_vkbuffer(vkapi::IVkBuffer buffer,
                                   size_t size,
                                   VkDeviceMemory memory,
                                   VkDeviceSize offset);

  DeviceAllocation import_vk_image(vkapi::IVkImage image,
                                   vkapi::IVkImageView view,
                                   VkImageLayout layout);

  vkapi::IVkImageView get_vk_imageview(const DeviceAllocation &alloc) const;

  vkapi::IVkImageView get_vk_lod_imageview(const DeviceAllocation &alloc,
                                           int lod) const;

  vkapi::IVkRenderPass get_renderpass(const VulkanRenderPassDesc &desc);

  vkapi::IVkFramebuffer get_framebuffer(const VulkanFramebufferDesc &desc);

  vkapi::IVkDescriptorSetLayout get_desc_set_layout(VulkanResourceSet &set);
  rhi_impl::RhiReturn<vkapi::IVkDescriptorSet> alloc_desc_set(
      vkapi::IVkDescriptorSetLayout layout);

  constexpr VulkanCapabilities &vk_caps() {
    return vk_caps_;
  }
  constexpr const VulkanCapabilities &vk_caps() const {
    return vk_caps_;
  }

  const VkPhysicalDeviceProperties &get_vk_physical_device_props() const {
    return vk_device_properties_;
  }

  // Profiler support
  void profiler_add_sampler(const std::string &kernel_name,
                            vkapi::IVkQueryPool query_pool) {
    samplers_.push_back(std::make_pair(kernel_name, query_pool));
  }

  vkapi::IVkQueryPool profiler_get_last_query_pool() {
    return samplers_.back().second;
  }

  size_t profiler_get_sampler_count() override {
    return samplers_.size();
  }

  void profiler_sync() override;
  std::vector<std::pair<std::string, double>> profiler_flush_sampled_time()
      override;

 private:
  friend VulkanSurface;

  void create_vma_allocator();
  [[nodiscard]] RhiResult new_descriptor_pool();

  VulkanCapabilities vk_caps_;
  VkPhysicalDeviceProperties vk_device_properties_;

  VkInstance instance_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VmaAllocator allocator_{nullptr};
  VmaAllocator allocator_export_{nullptr};

  VkQueue compute_queue_{VK_NULL_HANDLE};
  uint32_t compute_queue_family_index_{0};

  VkQueue graphics_queue_{VK_NULL_HANDLE};
  uint32_t graphics_queue_family_index_{0};

  struct ThreadLocalStreams;
  std::unique_ptr<ThreadLocalStreams> compute_streams_{nullptr};
  std::unique_ptr<ThreadLocalStreams> graphics_streams_{nullptr};

  // Memory allocation
  struct AllocationInternal {
    // Allocation info from VMA or set by `import_vkbuffer`
    VmaAllocationInfo alloc_info;
    // VkBuffer handle (reference counted)
    vkapi::IVkBuffer buffer{nullptr};
    // Buffer Device Address
    VkDeviceAddress addr{0};
    // If mapped, the currently mapped address
    void *mapped{nullptr};
    // Is the allocation external (imported) or not (VMA)
    bool external{false};
  };

  // Images / Image views
  struct ImageAllocInternal {
    bool external{false};
    VmaAllocationInfo alloc_info{};
    vkapi::IVkImage image{nullptr};
    vkapi::IVkImageView view{nullptr};
    std::vector<vkapi::IVkImageView> view_lods{};
  };

  // Since we use the pointer to AllocationInternal as the `alloc_id`,
  // **pointer stability** is important.
  rhi_impl::SyncedPtrStableObjectList<AllocationInternal> allocations_;
  rhi_impl::SyncedPtrStableObjectList<ImageAllocInternal> image_allocations_;

  // Renderpass
  unordered_map<VulkanRenderPassDesc,
                vkapi::IVkRenderPass,
                RenderPassDescHasher>
      renderpass_pools_;

  // Descriptors / Layouts / Pools
  unordered_map<VulkanResourceSet,
                vkapi::IVkDescriptorSetLayout,
                VulkanResourceSet::SetLayoutHasher,
                VulkanResourceSet::SetLayoutCmp>
      desc_set_layouts_;
  vkapi::IVkDescriptorPool desc_pool_{nullptr};

  // Internal implementaion functions
  inline static AllocationInternal &get_alloc_internal(
      const DeviceAllocation &alloc) {
    return *reinterpret_cast<AllocationInternal *>(alloc.alloc_id);
  }

  inline static ImageAllocInternal &get_image_alloc_internal(
      const DeviceAllocation &alloc) {
    return *reinterpret_cast<ImageAllocInternal *>(alloc.alloc_id);
  }

  RhiResult map_internal(AllocationInternal &alloc_int,
                         size_t offset,
                         size_t size,
                         void **mapped_ptr);

  // Profiler support
  std::vector<std::pair<std::string, vkapi::IVkQueryPool>> samplers_;
  std::vector<std::pair<std::string, double>> sampled_records_;
};

}  // namespace vulkan
}  // namespace taichi::lang
