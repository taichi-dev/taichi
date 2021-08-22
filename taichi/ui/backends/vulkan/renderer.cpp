#include "renderer.h"
#include "taichi/ui/utils/utils.h"

#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"
#include "taichi/ui/backends/vulkan/vulkan_cuda_interop.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Renderer::init(GLFWwindow *window, const AppConfig &config) {
  app_context_.init(window, config);
  swap_chain_.init(&app_context_);
  create_render_passes(); 

  create_semaphores();
  import_semaphores();
  
}

void Renderer::create_render_passes() {
  // for now we only have one pass.
   VulkanRenderPassDesc desc;
  desc.depth_attachment = swap_chain_.depth_format();
  printf("depth format: %d\n",swap_chain_.depth_format());
  desc.clear_depth = true;
  desc.color_attachments = {{buffer_format_ti_to_vk(swap_chain_.surface().image_format()),true}};
  
  VkRenderPass pass = app_context_.vulkan_device().get_renderpass(desc);
  render_passes_.push_back(pass);

}

void Renderer::clear_command_buffer_cache() {
  
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
  float aspect_ratio = swap_chain_.width() /
                       (float)swap_chain_.height();
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
}

void Renderer::recreate_swap_chain() {
  create_render_passes();
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

    //cuda_vk_semaphore_signal((CUexternalSemaphore)prev_draw_finished_cuda_);
  }
}

void Renderer::prepare_for_next_frame() {
  next_renderable_ = 0;
  if (app_context_.config.ti_arch == Arch::cuda) {
    //cuda_vk_semaphore_wait((CUexternalSemaphore)prev_draw_finished_cuda_);
  }
}

void Renderer::draw_frame(Gui *gui) {
  uint32_t image_index = 0;


  if (app_context_.config.ti_arch == Arch::cuda) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
    //cuda_vk_semaphore_signal((CUexternalSemaphore)this_draw_data_ready_cuda_);
  }

  VkCommandBuffer command_buffer;

  if (!gui->is_empty) {
    clear_command_buffer_cache();
  }

  std::unique_ptr<CommandList> cmd_list = app_context().vulkan_device().new_command_list({CommandListType::Graphics});
  bool color_clear = true;
  auto image = swap_chain_.surface().get_target_image();
  auto depth_image = swap_chain_.depth_allocation();
  cmd_list->begin_renderpass(0,0,swap_chain_.width(),swap_chain_.height(),1,&image,&color_clear,&depth_image,true);

    
  for (int i = 0; i < next_renderable_; ++i) {
    renderables_[i]->record_this_frame_commands(cmd_list.get());
  }

  gui->draw(cmd_list.get());
  cmd_list->end_renderpass();
  app_context_.vulkan_device().submit(cmd_list.get());

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

}  // namespace vulkan

TI_UI_NAMESPACE_END
