#pragma once
#include "taichi/rhi/vulkan/vulkan_api.h"
// FIXME: (penguinliong) Remove this.
#include "taichi/rhi/vulkan/vulkan_resource_binder.h"

namespace taichi::lang {
namespace vulkan {

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

class VulkanCommandList : public CommandList {
 public:
  VulkanCommandList(VulkanDevice *ti_device,
                    VulkanStream *stream,
                    vkapi::IVkCommandBuffer buffer);
  ~VulkanCommandList() override;

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

} // namespace vulkan
} // namespace taichi::lang
