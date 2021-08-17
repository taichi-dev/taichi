#include "particles.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

Particles::Particles(Renderer *renderer) {
  init_particles(renderer, 1);
}

void Particles::update_ubo(glm::vec3 color,
                           bool use_per_vertex_color,
                           float radius,
                           const Scene &scene) {
  UniformBufferObject ubo;
  ubo.scene = scene.current_ubo_;
  ubo.color = glm::vec4(color, 1);
  ubo.radius = radius;
  ubo.window_width = renderer_->swap_chain().swap_chain_extent().width;
  ubo.window_height = renderer_->swap_chain().swap_chain_extent().height;
  ubo.tan_half_fov = tan(glm::radians(scene.camera_.fov) / 2);
  ubo.use_per_vertex_color = use_per_vertex_color;

  MappedMemory mapped(
      app_context_->device(),
      uniform_buffer_memories_[renderer_->swap_chain().curr_image_index()],
      sizeof(ubo));
  memcpy(mapped.data, &ubo, sizeof(ubo));
}

void Particles::update_data(const ParticlesInfo &info, const Scene &scene) {
  if (info.renderable_info.vertices.matrix_rows != 3 ||
      info.renderable_info.vertices.matrix_cols != 1) {
    throw std::runtime_error("Particles vertices requres 3-d vector fields");
  }

  size_t correct_ssbo_size = scene.point_lights_.size() * sizeof(PointLight);
  if (config_.ssbo_size != correct_ssbo_size) {
    resize_storage_buffers(correct_ssbo_size);
  }
  {
    MappedMemory mapped(
        app_context_->device(),
        storage_buffer_memories_[renderer_->swap_chain().curr_image_index()],
        correct_ssbo_size);
    memcpy(mapped.data, scene.point_lights_.data(), correct_ssbo_size);
  }

  Renderable::update_data(info.renderable_info);

  update_ubo(info.color, info.renderable_info.per_vertex_color.valid,
             info.radius, scene);
}

void Particles::init_particles(Renderer *renderer, int vertices_count) {
  RenderableConfig config = {
      vertices_count,
      vertices_count,
      sizeof(UniformBufferObject),
      1,
      renderer->app_context().config.package_path +
          "/shaders/Particles_vk_vert.spv",
      renderer->app_context().config.package_path +
          "/shaders/Particles_vk_frag.spv",
      TopologyType::Points,
  };

  Renderable::init(config, renderer);
  Renderable::init_render_resources();
}

void Particles::create_descriptor_set_layout() {
  VkDescriptorSetLayoutBinding ubo_layout_binding{};
  ubo_layout_binding.binding = 0;
  ubo_layout_binding.descriptorCount = 1;
  ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  ubo_layout_binding.pImmutableSamplers = nullptr;
  ubo_layout_binding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutBinding ssbo_layout_binding{};
  ssbo_layout_binding.binding = 1;
  ssbo_layout_binding.descriptorCount = 1;
  ssbo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ssbo_layout_binding.pImmutableSamplers = nullptr;
  ssbo_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array<VkDescriptorSetLayoutBinding, 2> bindings = {ubo_layout_binding,
                                                          ssbo_layout_binding};
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(app_context_->device(), &layout_info, nullptr,
                                  &descriptor_set_layout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void Particles::create_descriptor_sets() {
  std::vector<VkDescriptorSetLayout> layouts(
      renderer_->swap_chain().chain_size(), descriptor_set_layout_);

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = renderer_->swap_chain().chain_size();
  alloc_info.pSetLayouts = layouts.data();

  descriptor_sets_.resize(renderer_->swap_chain().chain_size());

  if (vkAllocateDescriptorSets(app_context_->device(), &alloc_info,
                               descriptor_sets_.data()) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor sets!");
  }

  for (size_t i = 0; i < renderer_->swap_chain().chain_size(); i++) {
    VkDescriptorBufferInfo ubo_info{};
    ubo_info.buffer = uniform_buffers_[i];
    ubo_info.offset = 0;
    ubo_info.range = config_.ubo_size;

    VkDescriptorBufferInfo ssbo_info{};
    ssbo_info.buffer = storage_buffers_[i];
    ssbo_info.offset = 0;
    ssbo_info.range = config_.ssbo_size;

    std::array<VkWriteDescriptorSet, 2> descriptor_writes{};

    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstSet = descriptor_sets_[i];
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &ubo_info;

    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstSet = descriptor_sets_[i];
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pBufferInfo = &ssbo_info;

    vkUpdateDescriptorSets(app_context_->device(),
                           static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(), 0, nullptr);
  }
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
