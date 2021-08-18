#include "renderer.h"
#include "taichi/ui/utils/utils.h"

#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

void Renderer::init(GLFWwindow *window, const AppConfig &config) {
  app_context_.init(window, config);
  swap_chain_.init(&app_context_);
  create_render_passes();
  swap_chain_.create_framebuffers(render_passes_[0]);

  create_semaphores();
  import_semaphores();

  cached_command_buffers_.resize(swap_chain_.chain_size());
  for (int i = 0; i < swap_chain_.chain_size(); ++i) {
    cached_command_buffers_[i] = VK_NULL_HANDLE;
  }
}

void Renderer::create_render_passes() {
  // for now we only have one pass.
  VkRenderPass pass;
  create_render_pass(pass, swap_chain_.swap_chain_image_format(),
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, app_context_.device(),
                     app_context_.physical_device());
  render_passes_.push_back(pass);
}

void Renderer::clear_command_buffer_cache() {
  for (int i = 0; i < cached_command_buffers_.size(); ++i) {
    if (cached_command_buffers_[i] != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(app_context_.device(), app_context_.command_pool(),
                           1, &cached_command_buffers_[i]);
    }
    cached_command_buffers_[i] = VK_NULL_HANDLE;
  }
}

void Renderer::create_semaphores() {
  create_semaphore(prev_draw_finished_vk_, app_context_.device());
  create_semaphore(this_draw_data_ready_vk_, app_context_.device());
}

template <typename T>
std::unique_ptr<Renderable> get_new_renderable(Renderer *r) {
  return std::unique_ptr<Renderable>{new T(r)};
}

template <typename T>
T *Renderer::get_renderable_of_type() {
  if (next_renderable_ >= renderables_.size()) {
    renderables_.push_back(get_new_renderable<T>(this));
    clear_command_buffer_cache();
  } else if (dynamic_cast<T *>(renderables_[next_renderable_].get()) ==
             nullptr) {
    renderables_.insert(renderables_.begin() + next_renderable_,
                        get_new_renderable<T>(this));
    clear_command_buffer_cache();
  }

  if (T *t = dynamic_cast<T *>(renderables_[next_renderable_].get())) {
    return t;
  } else {
    throw std::runtime_error("Failed to Get Renderable.");
  }
}
void Renderer::set_background_color(const glm::vec3 &color) {
  background_color_ = color;
}

void Renderer::set_image(const SetImageInfo &info) {
  SetImage *s = get_renderable_of_type<SetImage>();
  s->update_data(info);
  next_renderable_ += 1;
}

void Renderer::triangles(const TrianglesInfo &info) {
  Triangles *triangles = get_renderable_of_type<Triangles>();
  triangles->update_data(info);
  next_renderable_ += 1;
}

void Renderer::lines(const LinesInfo &info) {
  Lines *lines = get_renderable_of_type<Lines>();
  lines->update_data(info);
  next_renderable_ += 1;
}

void Renderer::circles(const CirclesInfo &info) {
  Circles *circles = get_renderable_of_type<Circles>();
  circles->update_data(info);
  next_renderable_ += 1;
}

void Renderer::mesh(const MeshInfo &info, Scene *scene) {
  Mesh *mesh = get_renderable_of_type<Mesh>();
  mesh->update_data(info, *scene);
  next_renderable_ += 1;
}

void Renderer::particles(const ParticlesInfo &info, Scene *scene) {
  Particles *particles = get_renderable_of_type<Particles>();
  particles->update_data(info, *scene);
  next_renderable_ += 1;
}

void Renderer::scene(Scene *scene) {
  if (scene->point_lights_.size() == 0) {
    printf("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.swap_chain_extent().width /
                       (float)swap_chain_.swap_chain_extent().height;
  scene->update_ubo(aspect_ratio);
  for (int i = 0; i < scene->mesh_infos_.size(); ++i) {
    mesh(scene->mesh_infos_[i], scene);
  }
  for (int i = 0; i < scene->particles_infos_.size(); ++i) {
    particles(scene->particles_infos_[i], scene);
  }
  scene->mesh_infos_.clear();
  scene->particles_infos_.clear();
  scene->point_lights_.clear();
}

void Renderer::cleanup() {
  for (auto &renderable : renderables_) {
    renderable->cleanup();
  }
  vkDestroySemaphore(app_context_.device(), prev_draw_finished_vk_, nullptr);
  vkDestroySemaphore(app_context_.device(), this_draw_data_ready_vk_, nullptr);
  swap_chain_.cleanup();
  app_context_.cleanup();
}

void Renderer::cleanup_swap_chain() {
  clear_command_buffer_cache();
  for (auto &renderable : renderables_) {
    renderable->cleanup_swap_chain();
  }

  for (VkRenderPass pass : render_passes_) {
    vkDestroyRenderPass(app_context_.device(), pass, nullptr);
  }
  render_passes_.clear();

  swap_chain_.cleanup_swap_chain();
}

void Renderer::recreate_swap_chain() {
  create_render_passes();
  swap_chain_.recreate_swap_chain(render_passes_[0]);
  for (auto &renderable : renderables_) {
    renderable->recreate_swap_chain();
  }
}

void Renderer::import_semaphores() {
  if (app_context_.config.ti_arch == Arch::cuda) {
    prev_draw_finished_cuda_ = (uint64_t)cuda_vk_import_semaphore(
        prev_draw_finished_vk_, app_context_.device());
    this_draw_data_ready_cuda_ = (uint64_t)cuda_vk_import_semaphore(
        this_draw_data_ready_vk_, app_context_.device());

    cuda_vk_semaphore_signal((CUexternalSemaphore)prev_draw_finished_cuda_);
  }
}

void Renderer::prepare_for_next_frame() {
  swap_chain_.update_image_index();
  next_renderable_ = 0;
  if (app_context_.config.ti_arch == Arch::cuda) {
    cuda_vk_semaphore_wait((CUexternalSemaphore)prev_draw_finished_cuda_);
  }
}

void Renderer::draw_frame(Gui *gui) {
  uint32_t image_index = swap_chain_.curr_image_index();

  if (swap_chain_.images_in_flight()[image_index] != VK_NULL_HANDLE) {
    vkWaitForFences(app_context_.device(), 1,
                    &swap_chain_.images_in_flight()[image_index], VK_TRUE,
                    UINT64_MAX);
  }
  swap_chain_.images_in_flight()[image_index] =
      swap_chain_.in_flight_scenes()[swap_chain_.current_frame()];

  if (app_context_.config.ti_arch == Arch::cuda) {
    cuda_vk_semaphore_signal((CUexternalSemaphore)this_draw_data_ready_cuda_);
  }

  VkCommandBuffer command_buffer;

  if (!gui->is_empty) {
    clear_command_buffer_cache();
  }

  if (cached_command_buffers_[image_index] != VK_NULL_HANDLE) {
    command_buffer = cached_command_buffers_[image_index];
  } else {
    command_buffer = create_new_command_buffer(app_context_.command_pool(),
                                               app_context_.device());

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
      throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = render_passes_[0];
    render_pass_info.framebuffer =
        swap_chain_.swap_chain_framebuffers()[image_index];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = swap_chain_.swap_chain_extent();

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {
        {background_color_.x, background_color_.y, background_color_.z, 1.0f}};
    clear_values[1].depthStencil = {0.0f, 0};

    render_pass_info.clearValueCount =
        static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();
    vkCmdBeginRenderPass(command_buffer, &render_pass_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    for (int i = 0; i < next_renderable_; ++i) {
      renderables_[i]->record_this_frame_commands(command_buffer);
    }

    gui->draw(command_buffer);

    vkCmdEndRenderPass(command_buffer);
    vkEndCommandBuffer(command_buffer);

    cached_command_buffers_[image_index] = command_buffer;
  }

  std::vector<VkSemaphore> wait_semaphores = {
      swap_chain_.image_available_semaphores()[swap_chain_.current_frame()]};
  std::vector<VkPipelineStageFlags> wait_stages = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
  std::vector<VkSemaphore> signal_semaphores = {
      swap_chain_.render_finished_semaphores()[swap_chain_.current_frame()]};

  if (app_context_.config.ti_arch == Arch::cuda) {
    wait_semaphores.push_back(this_draw_data_ready_vk_);
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    signal_semaphores.push_back(prev_draw_finished_vk_);
  }

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  submit_info.waitSemaphoreCount = wait_semaphores.size();
  submit_info.pWaitSemaphores = wait_semaphores.data();
  submit_info.pWaitDstStageMask = wait_stages.data();

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  submit_info.signalSemaphoreCount = signal_semaphores.size();
  submit_info.pSignalSemaphores = signal_semaphores.data();

  vkResetFences(app_context_.device(), 1,
                &swap_chain_.in_flight_scenes()[swap_chain_.current_frame()]);

  if (vkQueueSubmit(
          app_context_.graphics_queue(), 1, &submit_info,
          swap_chain_.in_flight_scenes()[swap_chain_.current_frame()]) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to submit draw command buffer!");
  }
}

const std::vector<VkRenderPass> &Renderer::render_passes() const {
  return render_passes_;
}

const AppContext &Renderer::app_context() const {
  return app_context_;
}

AppContext &Renderer::app_context() {
  return app_context_;
}

const SwapChain &Renderer::swap_chain() const {
  return swap_chain_;
}

SwapChain &Renderer::swap_chain() {
  return swap_chain_;
}

void Renderer::present_frame() {
  swap_chain_.present_frame();
  if (swap_chain_.requires_recreate()) {
    recreate_swap_chain();
  }
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
