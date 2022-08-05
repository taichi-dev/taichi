#pragma once

#include "taichi/rhi/vulkan/vulkan_api.h"

#include <vk_mem_alloc.h>

#ifdef ANDROID
#include <android/native_window_jni.h>
#else
#include <GLFW/glfw3.h>
#endif

#include <memory>
#include <optional>

#include <taichi/rhi/device.h>
#include <taichi/rhi/vulkan/vulkan_utils.h>
#include <taichi/common/ref_counted_pool.h>

namespace taichi {
namespace lang {
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
    // TODO: Come up with a better hash
    size_t hash = 0;
    for (auto pair : desc.color_attachments) {
      hash ^= (size_t(pair.first) + pair.second);
      hash = (hash << 3) || (hash >> 61);
    }
    hash ^= (size_t(desc.depth_attachment) + desc.clear_depth);
    hash = (hash << 3) || (hash >> 61);
    return hash;
  }
};

struct VulkanFramebufferDesc {
  std::vector<vkapi::IVkImageView> attachments;
  uint32_t width;
  uint32_t height;
  vkapi::IVkRenderPass renderpass;

  bool operator==(const VulkanFramebufferDesc &other) const {
    return width == other.width && height == other.height &&
           renderpass == other.renderpass && attachments == other.attachments;
  }
};

struct FramebufferDescHasher {
  std::size_t operator()(const VulkanFramebufferDesc &desc) const {
    size_t hash = 0;
    for (auto view : desc.attachments) {
      hash ^= size_t(view->view);
      hash = (hash << 3) || (hash >> 61);
    }
    hash ^= desc.width;
    hash ^= desc.height;
    hash ^= size_t(desc.renderpass->renderpass);
    return hash;
  }
};

class VulkanResourceBinder : public ResourceBinder {
 public:
  struct Binding {
    VkDescriptorType type;
    DevicePtr ptr;
    VkDeviceSize size;
    union {
      VkSampler sampler{VK_NULL_HANDLE};  // used only for images
      int image_lod;
    };

    bool operator==(const Binding &other) const {
      return other.type == type && other.ptr == ptr && other.size == size &&
             other.sampler == sampler;
    }

    bool operator!=(const Binding &other) const {
      return !(*this == other);
    }
  };

  struct Set {
    std::unordered_map<uint32_t, Binding> bindings;

    // The compare function is for the hashmap to locate a set layout
    bool operator==(const Set &other) const {
      if (other.bindings.size() != bindings.size()) {
        return false;
      }
      for (auto &pair : bindings) {
        auto other_binding_iter = other.bindings.find(pair.first);
        if (other_binding_iter == other.bindings.end()) {
          return false;
        }
        const Binding &other_binding = other_binding_iter->second;
        if (other_binding.type != pair.second.type) {
          return false;
        }
      }
      return true;
    }

    bool operator!=(const Set &other) const {
      return !(*this == other);
    }
  };

  struct SetLayoutHasher {
    std::size_t operator()(const Set &set) const {
      // TODO: Come up with a better hash
      size_t hash = 0;
      for (const auto &pair : set.bindings) {
        hash = (hash ^ size_t(pair.second.type)) ^ size_t(pair.first);
      }
      return hash;
    }
  };

  struct DescSetCmp {
    bool operator()(const Set &a, const Set &b) const {
      if (a.bindings.size() != b.bindings.size()) {
        return false;
      }
      for (auto &pair : a.bindings) {
        auto other_binding_iter = b.bindings.find(pair.first);
        if (other_binding_iter == b.bindings.end()) {
          return false;
        }
        const Binding &other_binding = other_binding_iter->second;
        if (other_binding != pair.second) {
          return false;
        }
      }
      return true;
    }
  };

  struct DescSetHasher {
    std::size_t operator()(const Set &set) const {
      // TODO: Come up with a better hash
      size_t hash = 0;
      for (const auto &pair : set.bindings) {
        size_t binding_hash = 0;
        uint32_t *u32_ptr = (uint32_t *)&pair.second;
        static_assert(
            sizeof(VulkanResourceBinder::Binding) % sizeof(uint32_t) == 0,
            "sizeof(VulkanResourceBinder::Binding) is not a multiple of 4");
        size_t n = sizeof(VulkanResourceBinder::Binding) / sizeof(uint32_t);
        for (size_t i = 0; i < n; i++) {
          binding_hash = binding_hash ^ u32_ptr[i];
          binding_hash = (binding_hash << 7) | (binding_hash >> (64 - 7));
        }
        binding_hash = binding_hash ^ pair.first;
        binding_hash =
            (binding_hash << pair.first) | (binding_hash >> (64 - pair.first));
        hash = hash ^ binding_hash;
      }
      return hash;
    }
  };

  struct VulkanBindings : public Bindings {
    std::vector<
        std::pair<vkapi::IVkDescriptorSetLayout, vkapi::IVkDescriptorSet>>
        sets;
  };

  VulkanResourceBinder(
      VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_COMPUTE);
  ~VulkanResourceBinder();

  std::unique_ptr<Bindings> materialize() override;

  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DevicePtr ptr,
                 size_t size) override;
  void rw_buffer(uint32_t set,
                 uint32_t binding,
                 DeviceAllocation alloc) override;
  void buffer(uint32_t set,
              uint32_t binding,
              DevicePtr ptr,
              size_t size) override;
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc) override;
  void image(uint32_t set,
             uint32_t binding,
             DeviceAllocation alloc,
             ImageSamplerConfig sampler_config) override;
  void rw_image(uint32_t set,
                uint32_t binding,
                DeviceAllocation alloc,
                int lod) override;
  void vertex_buffer(DevicePtr ptr, uint32_t binding = 0) override;
  void index_buffer(DevicePtr ptr, size_t index_width) override;

  void write_to_set(uint32_t index,
                    VulkanDevice &device,
                    vkapi::IVkDescriptorSet set);
  Set &get_set(uint32_t index) {
    return sets_[index];
  }
  std::unordered_map<uint32_t, Set> &get_sets() {
    return sets_;
  }
  std::unordered_map<uint32_t, DevicePtr> &get_vertex_buffers() {
    return vertex_buffers_;
  }
  std::pair<DevicePtr, VkIndexType> get_index_buffer() {
    return std::make_pair(index_buffer_, index_type_);
  }

  void lock_layout();

 private:
  std::unordered_map<uint32_t, Set> sets_;
  bool layout_locked_{false};
  VkPipelineBindPoint bind_point_;

  std::unordered_map<uint32_t, DevicePtr> vertex_buffers_;
  DevicePtr index_buffer_{kDeviceNullPtr};
  VkIndexType index_type_;
};

// VulkanPipeline maps to a vkapi::IVkPipeline, or a SPIR-V module (a GLSL
// compute shader).
class VulkanPipeline : public Pipeline {
 public:
  struct Params {
    VulkanDevice *device{nullptr};
    std::vector<SpirvCodeView> code;
    std::string name{"Pipeline"};
  };

  explicit VulkanPipeline(const Params &params);
  explicit VulkanPipeline(
      const Params &params,
      const RasterParams &raster_params,
      const std::vector<VertexInputBinding> &vertex_inputs,
      const std::vector<VertexInputAttribute> &vertex_attrs);
  ~VulkanPipeline();

  ResourceBinder *resource_binder() override {
    return &resource_binder_;
  }

  vkapi::IVkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }

  vkapi::IVkPipeline pipeline() const {
    return pipeline_;
  }

  vkapi::IVkPipeline graphics_pipeline(
      const VulkanRenderPassDesc &renderpass_desc,
      vkapi::IVkRenderPass renderpass);

  const std::string &name() const {
    return name_;
  }

  bool is_graphics() const {
    return graphics_pipeline_template_ != nullptr;
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

  VkDevice device_{VK_NULL_HANDLE};  // not owned

  std::string name_;

  std::vector<VkPipelineShaderStageCreateInfo> shader_stages_;

  std::unique_ptr<GraphicsPipelineTemplate> graphics_pipeline_template_;
  std::unordered_map<vkapi::IVkRenderPass, vkapi::IVkPipeline>
      graphics_pipeline_;

  VulkanResourceBinder resource_binder_;
  std::vector<vkapi::IVkDescriptorSetLayout> set_layouts_;
  std::vector<VkShaderModule> shader_modules_;
  vkapi::IVkPipeline pipeline_{VK_NULL_HANDLE};
  vkapi::IVkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
};

class VulkanDeviceEvent : public DeviceEvent {
 public:
  VulkanDeviceEvent(vkapi::IVkEvent event) : vkapi_ref(event) {
  }
  ~VulkanDeviceEvent() {
  }

  vkapi::IVkEvent vkapi_ref{nullptr};
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device,
                    VulkanStream *stream,
                    vkapi::IVkCommandBuffer buffer);
  ~VulkanCommandList();

  void bind_pipeline(Pipeline *p) override;
  void bind_resources(ResourceBinder *binder) override;
  void bind_resources(ResourceBinder *binder,
                      ResourceBinder::Bindings *bindings) override;
  void buffer_barrier(DevicePtr ptr, size_t size) override;
  void buffer_barrier(DeviceAllocation alloc) override;
  void memory_barrier() override;
  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size) override;
  void buffer_fill(DevicePtr ptr, size_t size, uint32_t data) override;
  void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;
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

  void signal_event(DeviceEvent *event) override;
  void reset_event(DeviceEvent *event) override;
  void wait_event(DeviceEvent *event) override;

  vkapi::IVkRenderPass current_renderpass();

  // Vulkan specific functions
  vkapi::IVkCommandBuffer finalize();

  vkapi::IVkCommandBuffer vk_command_buffer();
  vkapi::IVkQueryPool vk_query_pool();

 private:
  bool finalized_{false};
  VulkanDevice *ti_device_;
  VulkanStream *stream_;
  VkDevice device_;
  vkapi::IVkQueryPool query_pool_;
  vkapi::IVkCommandBuffer buffer_;
  VulkanPipeline *current_pipeline_{nullptr};

  std::unordered_map<VulkanResourceBinder::Set,
                     vkapi::IVkDescriptorSet,
                     VulkanResourceBinder::DescSetHasher,
                     VulkanResourceBinder::DescSetCmp>
      currently_used_sets_;

  // Renderpass & raster pipeline
  VulkanRenderPassDesc current_renderpass_desc_;
  vkapi::IVkRenderPass current_renderpass_{VK_NULL_HANDLE};
  vkapi::IVkFramebuffer current_framebuffer_{VK_NULL_HANDLE};
  uint32_t viewport_width_{0}, viewport_height_{0};
};

class VulkanSurface : public Surface {
 public:
  VulkanSurface(VulkanDevice *device, const SurfaceConfig &config);
  ~VulkanSurface();

  StreamSemaphore acquire_next_image() override;
  DeviceAllocation get_target_image() override;

  void present_image(
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  std::pair<uint32_t, uint32_t> get_size() override;
  int get_image_count() override;
  BufferFormat image_format() override;
  void resize(uint32_t width, uint32_t height) override;

  DeviceAllocation get_depth_data(DeviceAllocation &depth_alloc) override;
  DeviceAllocation get_image_data() override;

 private:
  void create_swap_chain();
  void destroy_swap_chain();

  SurfaceConfig config_;

  VulkanDevice *device_;
  VkSurfaceKHR surface_;
  VkSwapchainKHR swapchain_;
  vkapi::IVkSemaphore image_available_;
#ifdef ANDROID
  ANativeWindow *window_;
#else
  GLFWwindow *window_;
#endif
  BufferFormat image_format_;

  uint32_t image_index_{0};

  std::vector<DeviceAllocation> swapchain_images_;

  // DeviceAllocation screenshot_image_{kDeviceNullAllocation};
  DeviceAllocation depth_buffer_{kDeviceNullAllocation};
  DeviceAllocation screenshot_buffer_{kDeviceNullAllocation};
};

struct DescPool {
  VkDescriptorPool pool;
  // Threads share descriptor sets
  RefCountedPool<vkapi::IVkDescriptorSet, true> sets;

  DescPool(VkDescriptorPool pool) : pool(pool) {
  }
};

class VulkanStreamSemaphoreObject : public StreamSemaphoreObject {
 public:
  VulkanStreamSemaphoreObject(vkapi::IVkSemaphore sema) : vkapi_ref(sema) {
  }
  ~VulkanStreamSemaphoreObject() {
  }

  vkapi::IVkSemaphore vkapi_ref{nullptr};
};

class VulkanStream : public Stream {
 public:
  VulkanStream(VulkanDevice &device,
               VkQueue queue,
               uint32_t queue_family_index);
  ~VulkanStream();

  std::unique_ptr<CommandList> new_command_list() override;
  StreamSemaphore submit(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;
  StreamSemaphore submit_synced(
      CommandList *cmdlist,
      const std::vector<StreamSemaphore> &wait_semaphores = {}) override;

  void command_sync() override;

  double device_time_elapsed_us() const override;

 private:
  struct TrackedCmdbuf {
    vkapi::IVkFence fence;
    vkapi::IVkCommandBuffer buf;
    vkapi::IVkQueryPool query_pool;
  };

  VulkanDevice &device_;
  VkQueue queue_;
  uint32_t queue_family_index_;

  // Command pools are per-thread
  vkapi::IVkCommandPool command_pool_;
  std::vector<TrackedCmdbuf> submitted_cmdbuffers_;
  double device_time_elapsed_us_;
};

class TI_DLL_EXPORT VulkanDevice : public GraphicsDevice {
 public:
  struct Params {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family_index;
    VkQueue graphics_queue;
    uint32_t graphics_queue_family_index;
  };

  VulkanDevice();
  void init_vulkan_structs(Params &params);
  ~VulkanDevice() override;

  std::unique_ptr<Pipeline> create_pipeline(
      const PipelineSourceDesc &src,
      std::string name = "Pipeline") override;
  std::unique_ptr<DeviceEvent> create_event() override;

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation handle) override;

  uint64_t get_memory_physical_pointer(DeviceAllocation handle) override;

  // Mapping can fail and will return nullptr
  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;

  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;

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

  std::tuple<vkapi::IVkImage, vkapi::IVkImageView, VkFormat> get_vk_image(
      const DeviceAllocation &alloc) const;

  DeviceAllocation import_vkbuffer(vkapi::IVkBuffer buffer);

  DeviceAllocation import_vk_image(vkapi::IVkImage image,
                                   vkapi::IVkImageView view,
                                   VkFormat format);

  vkapi::IVkImageView get_vk_imageview(const DeviceAllocation &alloc) const;

  vkapi::IVkImageView get_vk_lod_imageview(const DeviceAllocation &alloc,
                                           int lod) const;

  vkapi::IVkRenderPass get_renderpass(const VulkanRenderPassDesc &desc);

  vkapi::IVkFramebuffer get_framebuffer(const VulkanFramebufferDesc &desc);

  vkapi::IVkDescriptorSetLayout get_desc_set_layout(
      VulkanResourceBinder::Set &set);
  vkapi::IVkDescriptorSet alloc_desc_set(vkapi::IVkDescriptorSetLayout layout);

 private:
  friend VulkanSurface;

  void create_vma_allocator();
  void new_descriptor_pool();

  VkInstance instance_;
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  VmaAllocator allocator_;
  VmaAllocator allocator_export_{nullptr};

  VkQueue compute_queue_;
  uint32_t compute_queue_family_index_;

  VkQueue graphics_queue_;
  uint32_t graphics_queue_family_index_;

  struct ThreadLocalStreams;
  std::unique_ptr<ThreadLocalStreams> compute_streams_{nullptr};
  std::unique_ptr<ThreadLocalStreams> graphics_streams_{nullptr};

  // Memory allocation
  struct AllocationInternal {
    bool external{false};
    VmaAllocationInfo alloc_info;
    vkapi::IVkBuffer buffer;
    void *mapped{nullptr};
    VkDeviceAddress addr{0};
  };

  unordered_map<uint32_t, AllocationInternal> allocations_;

  uint32_t alloc_cnt_ = 0;

  // Images / Image views
  struct ImageAllocInternal {
    bool external{false};
    VmaAllocationInfo alloc_info;
    vkapi::IVkImage image;
    vkapi::IVkImageView view;
    std::vector<vkapi::IVkImageView> view_lods;
    VkFormat format;
  };

  unordered_map<uint32_t, ImageAllocInternal> image_allocations_;

  // Renderpass
  unordered_map<VulkanRenderPassDesc,
                vkapi::IVkRenderPass,
                RenderPassDescHasher>
      renderpass_pools_;
  unordered_map<VulkanFramebufferDesc,
                vkapi::IVkFramebuffer,
                FramebufferDescHasher>
      framebuffer_pools_;

  // Descriptors / Layouts / Pools
  unordered_map<VulkanResourceBinder::Set,
                vkapi::IVkDescriptorSetLayout,
                VulkanResourceBinder::SetLayoutHasher>
      desc_set_layouts_;
  vkapi::IVkDescriptorPool desc_pool_{nullptr};
};

VkFormat buffer_format_ti_to_vk(BufferFormat f);

BufferFormat buffer_format_vk_to_ti(VkFormat f);

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
