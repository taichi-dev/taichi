#pragma once

#include <volk.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <external/VulkanMemoryAllocator/include/vk_mem_alloc.h>

#include <GLFW/glfw3.h>

#include <memory>
#include <optional>

#include <taichi/backends/device.h>
#include <taichi/backends/vulkan/vulkan_utils.h>


namespace taichi {
namespace lang {
namespace vulkan {

class VulkanDevice;
class VulkanResourceBinder;

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
  std::vector<VkImageView> attachments;
  uint32_t width;
  uint32_t height;
  VkRenderPass renderpass;

  bool operator==(const VulkanFramebufferDesc &other) const {
    return width == other.width && height == other.height &&
           renderpass == other.renderpass && attachments == other.attachments;
  }
};

struct FramebufferDescHasher {
  std::size_t operator()(const VulkanFramebufferDesc &desc) const {
    size_t hash = 0;
    for (auto view : desc.attachments) {
      hash ^= size_t(view);
      hash = (hash << 3) || (hash >> 61);
    }
    hash ^= desc.width;
    hash ^= desc.height;
    hash ^= size_t(desc.renderpass);
    return hash;
  }
};

class VulkanResourceBinder : public ResourceBinder {
 public:
  struct Binding {
    VkDescriptorType type;
    DevicePtr ptr;
    size_t size;
  };

  struct Set {
    std::unordered_map<uint32_t, Binding> bindings;

    // The compare function is for the hashmap to locate a set layout
    bool operator==(const Set &other) const {
      if (other.bindings.size() != bindings.size()) {
        return false;
      }
      for (auto &pair : bindings) {
        const Binding &other_binding = other.bindings.at(pair.first);
        if (other_binding.type != pair.second.type) {
          return false;
        }
      }
      return true;
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

  VulkanResourceBinder(
      VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_COMPUTE);
  ~VulkanResourceBinder();
  void rw_buffer(uint32_t set, uint32_t binding, DevicePtr ptr, size_t size);
  void rw_buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc);
  void buffer(uint32_t set, uint32_t binding, DevicePtr ptr, size_t size);
  void buffer(uint32_t set, uint32_t binding, DeviceAllocation alloc);
  void vertex_buffer(DevicePtr ptr, uint32_t binding = 0);
  void index_buffer(DevicePtr ptr, size_t index_width);
  void framebuffer_color(DeviceAllocation image, uint32_t binding);
  void framebuffer_depth_stencil(DeviceAllocation image);

  void write_to_set(uint32_t index, VulkanDevice &device, VkDescriptorSet set);
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

// VulkanPipeline maps to a VkPipeline, or a SPIR-V module (a GLSL compute
// shader).
class VulkanPipeline : public Pipeline {
 public:
  struct Params {
    VulkanDevice *device{nullptr};
    std::vector<SpirvCodeView> code;
    std::string name{"Pipeline"};
  };

  struct RasterParams {
    VkPrimitiveTopology prim_topology{VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkCullModeFlagBits raster_cull_mode{VK_CULL_MODE_NONE};
    bool depth_test{false};
    bool depth_write{false};
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

  VkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }

  VkPipeline pipeline() const {
    return pipeline_;
  }

  VkPipeline graphics_pipeline(const VulkanRenderPassDesc &renderpass_desc,
                               VkRenderPass renderpass);

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
  std::unordered_map<VkRenderPass, VkPipeline> graphics_pipeline_;

  VulkanResourceBinder resource_binder_;
  std::vector<VkDescriptorSetLayout> set_layouts_;
  std::vector<VkShaderModule> shader_modules_;
  VkPipeline pipeline_{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
};

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device, VkCommandBuffer buffer,CommandListConfig config);
  ~VulkanCommandList();

  void bind_pipeline(Pipeline *p) override;
  void bind_resources(ResourceBinder *binder) override;
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
                        DeviceAllocation *depth_attachment,
                        bool depth_clear) override;
  void end_renderpass() override;
  void draw(uint32_t num_verticies, uint32_t start_vertex = 0) override;
  void draw_indexed(uint32_t num_indicies,
                    uint32_t start_vertex = 0,
                    uint32_t start_index = 0) override;

  // Vulkan specific functions
  VkCommandBuffer finalize();
  const CommandListConfig& config() const;

 private:
  CommandListConfig config_;

  bool finalized_{false};
  VulkanDevice *ti_device_;
  VkDevice device_;
  VkCommandBuffer buffer_;
  VulkanPipeline *current_pipeline_{nullptr};

  // Renderpass & raster pipeline
  VulkanRenderPassDesc current_renderpass_desc_;
  VkRenderPass current_renderpass_{VK_NULL_HANDLE};
  VkFramebuffer current_framebuffer_{VK_NULL_HANDLE};
  uint32_t viewport_width_{0}, viewport_height_{0};

  std::vector<std::pair<VkDescriptorSetLayout, VkDescriptorSet>> desc_sets_;
};

class VulkanSurface : public Surface {
 public:
  VulkanSurface(VulkanDevice *device,const SurfaceConfig& config);
  ~VulkanSurface();

  DeviceAllocation get_target_image() override;
  
  void present_image() override;
  std::pair<uint32_t, uint32_t> get_size() override;
  BufferFormat image_format() override;

 private:
  VulkanDevice *device_;
  VkSurfaceKHR surface_;
  VkSwapchainKHR swapchain_;
  VkSemaphore image_available_;
  GLFWwindow *window_;
  BufferFormat image_format_;

  uint32_t image_index_{0};

  std::vector<DeviceAllocation> swapchain_images_;
 };


struct VulkanMemoryPool {
  VmaPool pool;

  // the lifetime of these needs to == the lifetime of the vmapool.
  // because these are needed for allocating memory, which happens multiple times.
  VkExportMemoryAllocateInfoKHR export_mem_alloc_info{}; 
#ifdef _WIN64
  WindowsSecurityAttributes win_security_attribs;
  VkExportMemoryWin32HandleInfoKHR export_mem_win32_handle_info{};
#endif
};

class VulkanDevice : public GraphicsDevice {
 public:
  struct Params {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    VkCommandPool compute_pool;
    uint32_t compute_queue_family_index;
    VkQueue graphics_queue;
    VkCommandPool graphics_pool;
    uint32_t graphics_queue_family_index;
  };

  void init_vulkan_structs(Params &params);
  ~VulkanDevice() override;

  std::unique_ptr<Pipeline> create_pipeline(
      PipelineSourceDesc &src,
      std::string name = "Pipeline") override;

  DeviceAllocation allocate_memory(const AllocParams &params) override;
  void dealloc_memory(DeviceAllocation allocation) override;

  // Mapping can fail and will return nullptr
  void *map_range(DevicePtr ptr, uint64_t size) override;
  void *map(DeviceAllocation alloc) override;

  void unmap(DevicePtr ptr) override;
  void unmap(DeviceAllocation alloc) override;

  // Strictly intra device copy
  void memcpy_internal(DevicePtr dst, DevicePtr src, uint64_t size) override;

  std::unique_ptr<CommandList> new_command_list(CommandListConfig config) override;
  void dealloc_command_list(CommandList *cmdlist) override;
  void submit(CommandList *cmdlist) override;
  void submit_synced(CommandList *cmdlist) override;

  void command_sync() override;

  std::unique_ptr<Pipeline> create_raster_pipeline(
      std::vector<PipelineSourceDesc> &src,
      std::vector<BufferFormat> &render_target_formats,
      std::vector<VertexInputBinding> &vertex_inputs,
      std::vector<VertexInputAttribute> &vertex_attrs,
      std::string name = "Pipeline") override;

  std::unique_ptr<Surface> create_surface(const SurfaceConfig& config) override;

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

  VkCommandPool graphics_cmd_pool() const {
    return graphics_pool_;
  }

  VkCommandPool compute_cmd_pool() const {
    return compute_pool_;
  }

  std::tuple<VkDeviceMemory, size_t, size_t> get_vkmemory_offset_size(
      const DeviceAllocation &alloc) const;

  VkBuffer get_vkbuffer(const DeviceAllocation &alloc) const;

  std::tuple<VkImage, VkImageView, VkFormat> get_vk_image(
      const DeviceAllocation &alloc) const;
  DeviceAllocation import_vk_image(VkImage image, VkImageView view, VkFormat format);

  VkImageView get_vk_imageview(const DeviceAllocation &alloc) const;

  VkRenderPass get_renderpass(const VulkanRenderPassDesc &desc);

  VkFramebuffer get_framebuffer(const VulkanFramebufferDesc &desc);

  VkDescriptorSetLayout get_desc_set_layout(VulkanResourceBinder::Set &set);
  VkDescriptorSet alloc_desc_set(VkDescriptorSetLayout layout);
  void dealloc_desc_set(VkDescriptorSetLayout layout, VkDescriptorSet set);

 private:
  void create_vma_allocator();

  VkInstance instance_;
  VkDevice device_;
  VkPhysicalDevice physical_device_;
  VmaAllocator allocator_;
  VulkanMemoryPool export_pool_;
 

  VkQueue compute_queue_;
  VkCommandPool compute_pool_;
  uint32_t compute_queue_family_index_;

  VkQueue graphics_queue_;
  VkCommandPool graphics_pool_;
  uint32_t graphics_queue_family_index_;

  // Memory allocation
  struct AllocationInternal {
    VmaAllocation allocation;
    VmaAllocationInfo alloc_info;
    VkBuffer buffer;
    void *mapped{nullptr};
  };

  std::unordered_map<uint32_t, AllocationInternal> allocations_;

  uint32_t alloc_cnt_ = 0;

  // Images / Image views
  struct ImageAllocInternal {
    bool external{false};
    VmaAllocation allocation;
    VmaAllocationInfo alloc_info;
    VkImage image;
    VkImageView view;
    VkFormat format;
  };

  std::unordered_map<uint32_t, ImageAllocInternal> image_allocations_;

  // Command buffer tracking & allocation
  VkFence cmd_sync_fence_;

  std::unordered_multimap<VkCommandBuffer, VkFence> in_flight_cmdlists_;
  std::vector<VkCommandBuffer> dealloc_cmdlists_;
  std::vector<VkCommandBuffer> free_cmdbuffers_;

  // Renderpass
  std::unordered_map<VulkanRenderPassDesc, VkRenderPass, RenderPassDescHasher>
      renderpass_pools_;
  std::
      unordered_map<VulkanFramebufferDesc, VkFramebuffer, FramebufferDescHasher>
          framebuffer_pools_;

  // Descriptors / Layouts / Pools
  struct DescPool {
    VkDescriptorPool pool;
    std::vector<VkDescriptorSet> free_sets;
  };

  std::unordered_map<VulkanResourceBinder::Set,
                     VkDescriptorSetLayout,
                     VulkanResourceBinder::SetLayoutHasher>
      desc_set_layouts_;

  std::unordered_map<VkDescriptorSetLayout, DescPool> desc_set_pools_;

  std::unordered_multimap<VkDescriptorSet, VkFence> in_flight_desc_sets_;
  std::vector<std::pair<DescPool *, VkDescriptorSet>> dealloc_desc_sets_;
};


VkFormat buffer_format_ti_to_vk(BufferFormat f);

BufferFormat buffer_format_vk_to_ti(VkFormat f);

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
