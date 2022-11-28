#include "taichi/rhi/vulkan/vulkan_command_list.h"
#include "taichi/rhi/vulkan/vulkan_device.h"
#include "taichi/rhi/vulkan/vulkan_event.h"

namespace taichi::lang {
namespace vulkan {

VulkanCommandList::VulkanCommandList(VulkanDevice *ti_device,
                                     VulkanStream *stream,
                                     vkapi::IVkCommandBuffer buffer)
    : ti_device_(ti_device),
      stream_(stream),
      device_(ti_device->vk_device()),
#if !defined(__APPLE__)
      query_pool_(vkapi::create_query_pool(ti_device->vk_device())),
#else
      query_pool_(),
#endif
      buffer_(buffer) {
  VkCommandBufferBeginInfo info{};
  info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  info.pNext = nullptr;
  info.pInheritanceInfo = nullptr;
  info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  vkBeginCommandBuffer(buffer->buffer, &info);

// Workaround for MacOS: https://github.com/taichi-dev/taichi/issues/5888
#if !defined(__APPLE__)
  vkCmdResetQueryPool(buffer->buffer, query_pool_->query_pool, 0, 2);
  vkCmdWriteTimestamp(buffer->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                      query_pool_->query_pool, 0);
#endif
}

VulkanCommandList::~VulkanCommandList() {
}

void VulkanCommandList::bind_pipeline(Pipeline *p) {
  auto pipeline = static_cast<VulkanPipeline *>(p);

  if (current_pipeline_ == pipeline)
    return;

  if (pipeline->is_graphics()) {
    vkapi::IVkPipeline vk_pipeline = pipeline->graphics_pipeline(
        current_renderpass_desc_, current_renderpass_);
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vk_pipeline->pipeline);

    VkViewport viewport;
    viewport.width = viewport_width_;
    viewport.height = viewport_height_;
    viewport.x = 0;
    viewport.y = 0;
    viewport.minDepth = 0.0;
    viewport.maxDepth = 1.0;

    VkRect2D scissor;
    scissor.offset = {0, 0};
    scissor.extent = {viewport_width_, viewport_height_};

    vkCmdSetViewport(buffer_->buffer, 0, 1, &viewport);
    vkCmdSetScissor(buffer_->buffer, 0, 1, &scissor);
    vkCmdSetLineWidth(buffer_->buffer, 1.0f);
    buffer_->refs.push_back(vk_pipeline);
  } else {
    auto vk_pipeline = pipeline->pipeline();
    vkCmdBindPipeline(buffer_->buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      vk_pipeline->pipeline);
    buffer_->refs.push_back(vk_pipeline);
  }

  current_pipeline_ = pipeline;
}

void VulkanCommandList::bind_resources(ResourceBinder *ti_binder) {
  VulkanResourceBinder *binder = static_cast<VulkanResourceBinder *>(ti_binder);

  for (auto &pair : binder->get_sets()) {
    VkPipelineLayout pipeline_layout =
        current_pipeline_->pipeline_layout()->layout;

    vkapi::IVkDescriptorSetLayout layout =
        ti_device_->get_desc_set_layout(pair.second);

    vkapi::IVkDescriptorSet set = nullptr;

    if (currently_used_sets_.find(pair.second) != currently_used_sets_.end()) {
      set = currently_used_sets_.at(pair.second);
    }

    if (!set) {
      set = ti_device_->alloc_desc_set(layout);
      binder->write_to_set(pair.first, *ti_device_, set);
      currently_used_sets_[pair.second] = set;
    }

    VkPipelineBindPoint bind_point;
    if (current_pipeline_->is_graphics()) {
      bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
    } else {
      bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
    }

    vkCmdBindDescriptorSets(buffer_->buffer, bind_point, pipeline_layout,
                            /*firstSet=*/0,
                            /*descriptorSetCount=*/1, &set->set,
                            /*dynamicOffsetCount=*/0,
                            /*pDynamicOffsets=*/nullptr);
    buffer_->refs.push_back(set);
  }

  if (current_pipeline_->is_graphics()) {
    auto [idx_ptr, type] = binder->get_index_buffer();
    if (idx_ptr.device) {
      auto index_buffer = ti_device_->get_vkbuffer(idx_ptr);
      vkCmdBindIndexBuffer(buffer_->buffer, index_buffer->buffer,
                           idx_ptr.offset, type);
      buffer_->refs.push_back(index_buffer);
    }

    for (auto [binding, ptr] : binder->get_vertex_buffers()) {
      auto buffer = ti_device_->get_vkbuffer(ptr);
      vkCmdBindVertexBuffers(buffer_->buffer, binding, 1, &buffer->buffer,
                             &ptr.offset);
      buffer_->refs.push_back(buffer);
    }
  }
}

void VulkanCommandList::bind_resources(ResourceBinder *binder,
                                       ResourceBinder::Bindings *bindings) {
}

void VulkanCommandList::buffer_barrier(DevicePtr ptr, size_t size) {
  TI_ASSERT(ptr.device == ti_device_);

  auto buffer = ti_device_->get_vkbuffer(ptr);

  VkBufferMemoryBarrier barrier;
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.buffer = buffer->buffer;
  barrier.offset = ptr.offset;
  barrier.size = size;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_->buffer,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/0, nullptr,
      /*bufferMemoryBarrierCount=*/1,
      /*pBufferMemoryBarriers=*/&barrier,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::buffer_barrier(DeviceAllocation alloc) {
  buffer_barrier(DevicePtr{alloc, 0}, VK_WHOLE_SIZE);
}

void VulkanCommandList::memory_barrier() {
  VkMemoryBarrier barrier;
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
  barrier.dstAccessMask =
      (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

  vkCmdPipelineBarrier(
      buffer_->buffer,
      /*srcStageMask=*/
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*dstStageMask=*/VK_PIPELINE_STAGE_TRANSFER_BIT |
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      /*srcStageMask=*/0, /*memoryBarrierCount=*/1, &barrier,
      /*bufferMemoryBarrierCount=*/0,
      /*pBufferMemoryBarriers=*/nullptr,
      /*imageMemoryBarrierCount=*/0,
      /*pImageMemoryBarriers=*/nullptr);
}

void VulkanCommandList::buffer_copy(DevicePtr dst, DevicePtr src, size_t size) {
  VkBufferCopy copy_region{};
  copy_region.srcOffset = src.offset;
  copy_region.dstOffset = dst.offset;
  copy_region.size = size;
  auto src_buffer = ti_device_->get_vkbuffer(src);
  auto dst_buffer = ti_device_->get_vkbuffer(dst);
  vkCmdCopyBuffer(buffer_->buffer, src_buffer->buffer, dst_buffer->buffer,
                  /*regionCount=*/1, &copy_region);
  buffer_->refs.push_back(src_buffer);
  buffer_->refs.push_back(dst_buffer);
}

void VulkanCommandList::buffer_fill(DevicePtr ptr, size_t size, uint32_t data) {
  auto buffer = ti_device_->get_vkbuffer(ptr);
  vkCmdFillBuffer(buffer_->buffer, buffer->buffer, ptr.offset,
                  (size == kBufferSizeEntireSize) ? VK_WHOLE_SIZE : size, data);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::dispatch(uint32_t x, uint32_t y, uint32_t z) {
  vkCmdDispatch(buffer_->buffer, x, y, z);
}

vkapi::IVkCommandBuffer VulkanCommandList::vk_command_buffer() {
  return buffer_;
}

vkapi::IVkQueryPool VulkanCommandList::vk_query_pool() {
  return query_pool_;
}

void VulkanCommandList::begin_renderpass(int x0,
                                         int y0,
                                         int x1,
                                         int y1,
                                         uint32_t num_color_attachments,
                                         DeviceAllocation *color_attachments,
                                         bool *color_clear,
                                         std::vector<float> *clear_colors,
                                         DeviceAllocation *depth_attachment,
                                         bool depth_clear) {
  VulkanRenderPassDesc &rp_desc = current_renderpass_desc_;
  current_renderpass_desc_.color_attachments.clear();
  rp_desc.clear_depth = depth_clear;

  bool has_depth = false;

  if (depth_attachment) {
    auto [image, view, format] = ti_device_->get_vk_image(*depth_attachment);
    rp_desc.depth_attachment = format;
    has_depth = true;
  } else {
    rp_desc.depth_attachment = VK_FORMAT_UNDEFINED;
  }

  std::vector<VkClearValue> clear_values(num_color_attachments +
                                         (has_depth ? 1 : 0));

  VulkanFramebufferDesc fb_desc;

  for (uint32_t i = 0; i < num_color_attachments; i++) {
    auto [image, view, format] = ti_device_->get_vk_image(color_attachments[i]);
    rp_desc.color_attachments.emplace_back(format, color_clear[i]);
    fb_desc.attachments.push_back(view);
    clear_values[i].color =
        VkClearColorValue{{clear_colors[i][0], clear_colors[i][1],
                           clear_colors[i][2], clear_colors[i][3]}};
  }

  if (has_depth) {
    auto [depth_image, depth_view, depth_format] =
        ti_device_->get_vk_image(*depth_attachment);
    clear_values[num_color_attachments].depthStencil =
        VkClearDepthStencilValue{0.0, 0};
    fb_desc.attachments.push_back(depth_view);
  }

  current_renderpass_ = ti_device_->get_renderpass(rp_desc);

  fb_desc.width = x1 - x0;
  fb_desc.height = y1 - y0;
  fb_desc.renderpass = current_renderpass_;

  viewport_width_ = fb_desc.width;
  viewport_height_ = fb_desc.height;

  current_framebuffer_ = ti_device_->get_framebuffer(fb_desc);

  VkRect2D render_area;
  render_area.offset = {x0, y0};
  render_area.extent = {uint32_t(x1 - x0), uint32_t(y1 - y0)};

  VkRenderPassBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  begin_info.pNext = nullptr;
  begin_info.renderPass = current_renderpass_->renderpass;
  begin_info.framebuffer = current_framebuffer_->framebuffer;
  begin_info.renderArea = render_area;
  begin_info.clearValueCount = clear_values.size();
  begin_info.pClearValues = clear_values.data();

  vkCmdBeginRenderPass(buffer_->buffer, &begin_info,
                       VK_SUBPASS_CONTENTS_INLINE);
  buffer_->refs.push_back(current_renderpass_);
  buffer_->refs.push_back(current_framebuffer_);
}

void VulkanCommandList::end_renderpass() {
  vkCmdEndRenderPass(buffer_->buffer);

  current_renderpass_ = VK_NULL_HANDLE;
  current_framebuffer_ = VK_NULL_HANDLE;
}

void VulkanCommandList::draw(uint32_t num_verticies, uint32_t start_vertex) {
  vkCmdDraw(buffer_->buffer, num_verticies, /*instanceCount=*/1, start_vertex,
            /*firstInstance=*/0);
}

void VulkanCommandList::draw_instance(uint32_t num_verticies,
                                      uint32_t num_instances,
                                      uint32_t start_vertex,
                                      uint32_t start_instance) {
  vkCmdDraw(buffer_->buffer, num_verticies, num_instances, start_vertex,
            start_instance);
}

void VulkanCommandList::draw_indexed(uint32_t num_indicies,
                                     uint32_t start_vertex,
                                     uint32_t start_index) {
  vkCmdDrawIndexed(buffer_->buffer, num_indicies, /*instanceCount=*/1,
                   start_index, start_vertex,
                   /*firstInstance=*/0);
}

void VulkanCommandList::draw_indexed_instance(uint32_t num_indicies,
                                              uint32_t num_instances,
                                              uint32_t start_vertex,
                                              uint32_t start_index,
                                              uint32_t start_instance) {
  vkCmdDrawIndexed(buffer_->buffer, num_indicies, num_instances, start_index,
                   start_vertex, start_instance);
}

void VulkanCommandList::image_transition(DeviceAllocation img,
                                         ImageLayout old_layout_,
                                         ImageLayout new_layout_) {
  auto [image, view, format] = ti_device_->get_vk_image(img);

  VkImageLayout old_layout = image_layout_ti_to_vk(old_layout_);
  VkImageLayout new_layout = image_layout_ti_to_vk(new_layout_);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image->image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;

  static std::unordered_map<VkImageLayout, VkPipelineStageFlagBits> stages;
  stages[VK_IMAGE_LAYOUT_UNDEFINED] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_GENERAL] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = VK_PIPELINE_STAGE_TRANSFER_BIT;
  stages[VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL] = VK_PIPELINE_STAGE_TRANSFER_BIT;
  stages[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] =
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  stages[VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL] =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  stages[VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL] =
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  stages[VK_IMAGE_LAYOUT_PRESENT_SRC_KHR] = VK_PIPELINE_STAGE_TRANSFER_BIT;

  static std::unordered_map<VkImageLayout, VkAccessFlagBits> access;
  access[VK_IMAGE_LAYOUT_UNDEFINED] = (VkAccessFlagBits)0;
  access[VK_IMAGE_LAYOUT_GENERAL] =
      VkAccessFlagBits(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  access[VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL] = VK_ACCESS_TRANSFER_WRITE_BIT;
  access[VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL] = VK_ACCESS_TRANSFER_READ_BIT;
  access[VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL] = VK_ACCESS_MEMORY_READ_BIT;
  access[VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL] =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  access[VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL] =
      VkAccessFlagBits(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  access[VK_IMAGE_LAYOUT_PRESENT_SRC_KHR] = VK_ACCESS_MEMORY_READ_BIT;

  if (stages.find(old_layout) == stages.end() ||
      stages.find(new_layout) == stages.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  source_stage = stages.at(old_layout);
  destination_stage = stages.at(new_layout);

  if (access.find(old_layout) == access.end() ||
      access.find(new_layout) == access.end()) {
    throw std::invalid_argument("unsupported layout transition!");
  }
  barrier.srcAccessMask = access.at(old_layout);
  barrier.dstAccessMask = access.at(new_layout);

  vkCmdPipelineBarrier(buffer_->buffer, source_stage, destination_stage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);
  buffer_->refs.push_back(image);
}

inline void buffer_image_copy_ti_to_vk(VkBufferImageCopy &copy_info,
                                       size_t offset,
                                       const BufferImageCopyParams &params) {
  copy_info.bufferOffset = offset;
  copy_info.bufferRowLength = params.buffer_row_length;
  copy_info.bufferImageHeight = params.buffer_image_height;
  copy_info.imageExtent.width = params.image_extent.x;
  copy_info.imageExtent.height = params.image_extent.y;
  copy_info.imageExtent.depth = params.image_extent.z;
  copy_info.imageOffset.x = params.image_offset.x;
  copy_info.imageOffset.y = params.image_offset.y;
  copy_info.imageOffset.z = params.image_offset.z;
  copy_info.imageSubresource.aspectMask =
      params.image_aspect_flag;  // FIXME: add option in BufferImageCopyParams
                                 // to support copying depth images
                                 // FIXED: added an option in
                                 // BufferImageCopyParams as image_aspect_flag
                                 // by yuhaoLong(mocki)
  copy_info.imageSubresource.baseArrayLayer = params.image_base_layer;
  copy_info.imageSubresource.layerCount = params.image_layer_count;
  copy_info.imageSubresource.mipLevel = params.image_mip_level;
}

void VulkanCommandList::buffer_to_image(DeviceAllocation dst_img,
                                        DevicePtr src_buf,
                                        ImageLayout img_layout,
                                        const BufferImageCopyParams &params) {
  VkBufferImageCopy copy_info{};
  buffer_image_copy_ti_to_vk(copy_info, src_buf.offset, params);

  auto [image, view, format] = ti_device_->get_vk_image(dst_img);
  auto buffer = ti_device_->get_vkbuffer(src_buf);

  vkCmdCopyBufferToImage(buffer_->buffer, buffer->buffer, image->image,
                         image_layout_ti_to_vk(img_layout), 1, &copy_info);
  buffer_->refs.push_back(image);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::image_to_buffer(DevicePtr dst_buf,
                                        DeviceAllocation src_img,
                                        ImageLayout img_layout,
                                        const BufferImageCopyParams &params) {
  VkBufferImageCopy copy_info{};
  buffer_image_copy_ti_to_vk(copy_info, dst_buf.offset, params);

  auto [image, view, format] = ti_device_->get_vk_image(src_img);
  auto buffer = ti_device_->get_vkbuffer(dst_buf);

  vkCmdCopyImageToBuffer(buffer_->buffer, image->image,
                         image_layout_ti_to_vk(img_layout), buffer->buffer, 1,
                         &copy_info);
  buffer_->refs.push_back(image);
  buffer_->refs.push_back(buffer);
}

void VulkanCommandList::copy_image(DeviceAllocation dst_img,
                                   DeviceAllocation src_img,
                                   ImageLayout dst_img_layout,
                                   ImageLayout src_img_layout,
                                   const ImageCopyParams &params) {
  VkImageCopy copy{};
  copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.srcSubresource.layerCount = 1;
  copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  copy.dstSubresource.layerCount = 1;
  copy.extent.width = params.width;
  copy.extent.height = params.height;
  copy.extent.depth = params.depth;

  auto [dst_vk_image, dst_view, dst_format] = ti_device_->get_vk_image(dst_img);
  auto [src_vk_image, src_view, src_format] = ti_device_->get_vk_image(src_img);

  vkCmdCopyImage(buffer_->buffer, src_vk_image->image,
                 image_layout_ti_to_vk(src_img_layout), dst_vk_image->image,
                 image_layout_ti_to_vk(dst_img_layout), 1, &copy);

  buffer_->refs.push_back(dst_vk_image);
  buffer_->refs.push_back(src_vk_image);
}

void VulkanCommandList::blit_image(DeviceAllocation dst_img,
                                   DeviceAllocation src_img,
                                   ImageLayout dst_img_layout,
                                   ImageLayout src_img_layout,
                                   const ImageCopyParams &params) {
  VkOffset3D blit_size;
  blit_size.x = params.width;
  blit_size.y = params.height;
  blit_size.z = params.depth;
  VkImageBlit blit{};
  blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.srcSubresource.layerCount = 1;
  blit.srcOffsets[1] = blit_size;
  blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  blit.dstSubresource.layerCount = 1;
  blit.dstOffsets[1] = blit_size;

  auto [dst_vk_image, dst_view, dst_format] = ti_device_->get_vk_image(dst_img);
  auto [src_vk_image, src_view, src_format] = ti_device_->get_vk_image(src_img);

  vkCmdBlitImage(buffer_->buffer, src_vk_image->image,
                 image_layout_ti_to_vk(src_img_layout), dst_vk_image->image,
                 image_layout_ti_to_vk(dst_img_layout), 1, &blit,
                 VK_FILTER_NEAREST);

  buffer_->refs.push_back(dst_vk_image);
  buffer_->refs.push_back(src_vk_image);
}

void VulkanCommandList::signal_event(DeviceEvent *event) {
  VulkanDeviceEvent *event2 = static_cast<VulkanDeviceEvent *>(event);
  vkCmdSetEvent(buffer_->buffer, event2->vkapi_ref->event,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
void VulkanCommandList::reset_event(DeviceEvent *event) {
  VulkanDeviceEvent *event2 = static_cast<VulkanDeviceEvent *>(event);
  vkCmdResetEvent(buffer_->buffer, event2->vkapi_ref->event,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
void VulkanCommandList::wait_event(DeviceEvent *event) {
  VulkanDeviceEvent *event2 = static_cast<VulkanDeviceEvent *>(event);
  vkCmdWaitEvents(buffer_->buffer, 1, &event2->vkapi_ref->event,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0,
                  nullptr);
}

void VulkanCommandList::set_line_width(float width) {
  if (ti_device_->vk_caps().wide_line) {
    vkCmdSetLineWidth(buffer_->buffer, width);
  }
}

vkapi::IVkRenderPass VulkanCommandList::current_renderpass() {
  return current_renderpass_;
}

vkapi::IVkCommandBuffer VulkanCommandList::finalize() {
  if (!finalized_) {
// Workaround for MacOS: https://github.com/taichi-dev/taichi/issues/5888
#if !defined(__APPLE__)
    vkCmdWriteTimestamp(buffer_->buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        query_pool_->query_pool, 1);
#endif
    vkEndCommandBuffer(buffer_->buffer);
    finalized_ = true;
  }
  return buffer_;
}

}  // namespace vulkan
}  // namespace taichi::lang
