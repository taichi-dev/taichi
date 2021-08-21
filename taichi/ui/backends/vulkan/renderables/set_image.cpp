#include "set_image.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "kernels.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

void SetImage::update_data(const SetImageInfo &info) {
  const FieldInfo &img = info.img;
  if (img.shape.size() != 2) {
    throw std::runtime_error(
        "for set image, the image should have exactly two axis. e,g, "
        "ti.Vector.field(3,ti.u8,(1920,1080) ");
  }
  if ((img.matrix_rows != 3 && img.matrix_rows != 4) || img.matrix_cols != 1) {
    throw std::runtime_error(
        "for set image, the image should either a 3-D vector field (RGB) or a "
        "4D vector field (RGBA) ");
  }
  int new_width = img.shape[0];
  int new_height = img.shape[1];

  if (new_width != width || new_height != height) {
    cleanup_swap_chain();
    cleanup();
    init_set_image(renderer_, new_width, new_height);
  }

  int actual_width = next_power_of_2(width);
  int actual_height = next_power_of_2(height);

  int pixels = width * height;

  VkImage texture_image = std::get<0>(renderer_->app_context().vulkan_device().get_vk_image(texture_));

  if (img.field_source == FieldSource::TaichiCuda) {
    if (img.dtype == PrimitiveType::u8) {
      copy_to_texture_fuffer_cuda((unsigned char *)img.data,
                                  (uint64_t)texture_surface_, width, height,
                                  actual_width, actual_height, img.matrix_rows);
    } else if (img.dtype == PrimitiveType::f32) {
      copy_to_texture_fuffer_cuda((float *)img.data, (uint64_t)texture_surface_,
                                  width, height, actual_width, actual_height,
                                  img.matrix_rows);
    } else {
      throw std::runtime_error("for set image, dtype must be u8 or f32");
    }
  } else if (img.field_source == FieldSource::TaichiX64) {
    transition_image_layout(
        texture_image, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, app_context_->command_pool(),
        app_context_->device(), app_context_->graphics_queue());

    unsigned char* mapped = (unsigned char*)app_context_->vulkan_device().map(staging_buffer_);

    if (img.dtype == PrimitiveType::u8) {
      copy_to_texture_fuffer_x64(
          (unsigned char *)img.data, mapped, width,
          height, actual_width, actual_height, img.matrix_rows);
    } else if (img.dtype == PrimitiveType::f32) {
      copy_to_texture_fuffer_x64(
          (float *)img.data, mapped, width, height,
          actual_width, actual_height, img.matrix_rows);
    } else {
      throw std::runtime_error("for set image, dtype must be u8 or f32");
    }

    app_context_->vulkan_device().unmap(staging_buffer_);

    copy_buffer_to_image(app_context_->vulkan_device().get_vkbuffer(staging_buffer_), texture_image, width, height,
                         app_context_->command_pool(), app_context_->device(),
                         app_context_->graphics_queue());

    transition_image_layout(
        texture_image, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, app_context_->command_pool(),
        app_context_->device(), app_context_->graphics_queue());
  } else {
    throw std::runtime_error("unsupported field source");
  }
}

SetImage::SetImage(Renderer *renderer) {
  init_set_image(renderer, 1, 1);
}

void SetImage::init_set_image(Renderer *renderer,
                              int img_width,
                              int img_height) {
  RenderableConfig config = {
      6,
      6,
      0,
      0,
      renderer->app_context().config.package_path +
          "/shaders/SetImage_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/SetImage_vk_frag.spv",
      TopologyType::Triangles,
  };

  Renderable::init(config, renderer);

  width = img_width;
  height = img_height;

  create_texture();
  create_texture_sampler();

  Renderable::init_render_resources();

  update_vertex_buffer_();
  update_index_buffer_();
}


void SetImage::create_texture() {
  VkImage texture_image;
  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
  size_t image_size = width*height*4;

  create_image(3, width, height, 1, format,
               VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image,
               texture_image_memory_, app_context_->device(),
               app_context_->physical_device());

  transition_image_layout(
      texture_image, format, VK_IMAGE_LAYOUT_UNDEFINED,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, app_context_->command_pool(),
      app_context_->device(), app_context_->graphics_queue());
  transition_image_layout(texture_image, format,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          app_context_->command_pool(), app_context_->device(),
                          app_context_->graphics_queue());

  if (app_context_->config.ti_arch == Arch::cuda) {
    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(app_context_->device(), texture_image,
                                 &mem_requirements);

    auto handle =
        get_device_mem_handle(texture_image_memory_, app_context_->device());
    CUexternalMemory external_mem = import_vk_memory_object_from_handle(
        handle, mem_requirements.size, true);

    texture_surface_ = (uint64_t)get_image_surface_object_of_external_memory(
        external_mem, width, height, 1);
  }

  Device::AllocParams staging_buffer_params {image_size,true,false,false,AllocUsage::Uniform};
  staging_buffer_ = renderer_->app_context().vulkan_device().allocate_memory(staging_buffer_params);

  VkImageView view = create_image_view(3, texture_image, VK_FORMAT_R8G8B8A8_UNORM,
                        VK_IMAGE_ASPECT_COLOR_BIT, app_context_->device());

  texture_ =  renderer_->app_context().vulkan_device().import_vk_image(texture_image,view,format);
}



void SetImage::create_texture_sampler() {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(app_context_->physical_device(), &properties);

  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = VK_FILTER_LINEAR;
  sampler_info.minFilter = VK_FILTER_LINEAR;
  sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sampler_info.anisotropyEnable = VK_FALSE;
  sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(app_context_->device(), &sampler_info, nullptr,
                      &texture_sampler_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

void SetImage::update_vertex_buffer_() {
  const std::vector<Vertex> vertices = {
      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{-1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},

      {{-1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 1.f}, {1.f, 1.f, 1.f}},
      {{1.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f}, {1.f, 1.f, 1.f}},
      {{1.f, -1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f}, {1.f, 1.f, 1.f}},
  };

  {
    Vertex* mapped_vbo = (Vertex*)app_context_->vulkan_device().map(staging_vertex_buffer_);
      
    memcpy(mapped_vbo, vertices.data(),
           (size_t)config_.vertices_count * sizeof(Vertex));
    app_context_->vulkan_device().unmap(staging_vertex_buffer_);
  }

  app_context_->vulkan_device().memcpy(vertex_buffer_.get_ptr(0),staging_vertex_buffer_.get_ptr(0),config_.vertices_count * sizeof(Vertex));
  
}

void SetImage::update_index_buffer_() {
  const std::vector<uint32_t> indices = {
      0, 1, 2, 3, 4, 5,
  };
  {
    int* mapped_ibo = (int*)app_context_->vulkan_device().map(staging_index_buffer_);
    memcpy(mapped_ibo, indices.data(),
           (size_t)config_.indices_count * sizeof(int));
    app_context_->vulkan_device().unmap(staging_index_buffer_);
  }

  app_context_->vulkan_device().memcpy(index_buffer_.get_ptr(0),staging_index_buffer_.get_ptr(0),config_.indices_count * sizeof(int));

  indexed_ = true;
}

void SetImage::create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding sampler_layout_binding{};
  sampler_layout_binding.binding = 1;
  sampler_layout_binding.descriptorCount = 1;
  sampler_layout_binding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sampler_layout_binding.pImmutableSamplers = nullptr;
  sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array<VkDescriptorSetLayoutBinding, 1> bindings = {
      sampler_layout_binding};
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(app_context_->device(), &layout_info, nullptr,
                                  &descriptor_set_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void SetImage::create_descriptor_sets() {
  std::vector<VkDescriptorSetLayout> layouts(   1, descriptor_set_layout_);

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount =1;
  alloc_info.pSetLayouts = layouts.data();

  

  if (vkAllocateDescriptorSets(app_context_->device(), &alloc_info,
                               &descriptor_set_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  
    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = std::get<1>(renderer_->app_context().vulkan_device().get_vk_image(texture_));
    image_info.sampler = texture_sampler_;

    std::array<VkWriteDescriptorSet, 1> descriptor_writes{};

    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_set_;
    descriptor_writes[0].dstBinding = 1;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pImageInfo = &image_info;

    vkUpdateDescriptorSets(app_context_->device(),
                           static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(), 0, nullptr);
  
}

void SetImage::cleanup() {
  Renderable::cleanup();

  vkDestroySampler(app_context_->device(), texture_sampler_, nullptr);
  
  vkFreeMemory(app_context_->device(), texture_image_memory_, nullptr);

}

}  // namespace vulkan

TI_UI_NAMESPACE_END
