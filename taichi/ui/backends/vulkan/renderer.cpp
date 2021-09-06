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
}

template <typename T>
std::unique_ptr<Renderable> get_new_renderable(AppContext *app_context) {
  return std::unique_ptr<Renderable>{new T(app_context)};
}

template <typename T>
T *Renderer::get_renderable_of_type() {
  if (next_renderable_ >= renderables_.size()) {
    renderables_.push_back(get_new_renderable<T>(&app_context_));
  } else if (dynamic_cast<T *>(renderables_[next_renderable_].get()) ==
             nullptr) {
    renderables_.insert(renderables_.begin() + next_renderable_,
                        get_new_renderable<T>(&app_context_));
  }

  if (T *t = dynamic_cast<T *>(renderables_[next_renderable_].get())) {
    return t;
  } else {
    TI_ERROR("Failed to Get Renderable.");
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
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.width() / (float)swap_chain_.height();
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
  renderables_.clear();
  swap_chain_.cleanup();
  app_context_.cleanup();
}

void Renderer::prepare_for_next_frame() {
  next_renderable_ = 0;
}

void Renderer::draw_frame(Gui *gui) {
  uint32_t image_index = 0;

  if (app_context_.config.ti_arch == Arch::cuda) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }

  auto stream = app_context_.device().get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  bool color_clear = true;
  std::vector<float> clear_colors = {background_color_[0], background_color_[1],
                                     background_color_[2], 1};
  auto image = swap_chain_.surface().get_target_image();
  auto depth_image = swap_chain_.depth_allocation();
  cmd_list->begin_renderpass(
      /*xmin=*/0, /*ymin=*/0, /*xmax=*/swap_chain_.width(),
      /*ymax=*/swap_chain_.height(), /*num_color_attachments=*/1, &image,
      &color_clear, &clear_colors, &depth_image,
      /*depth_clear=*/true);

  for (int i = 0; i < next_renderable_; ++i) {
    renderables_[i]->record_this_frame_commands(cmd_list.get());
  }

  VkRenderPass pass = static_cast<VulkanCommandList *>(cmd_list.get())
                          ->current_renderpass()
                          ->renderpass;

  if (gui->render_pass() == VK_NULL_HANDLE) {
    gui->init_render_resources(pass);
  } else if (gui->render_pass() != pass) {
    gui->cleanup_render_resources();
    gui->init_render_resources(pass);
  }

  gui->draw(cmd_list.get());
  cmd_list->end_renderpass();
  stream->submit_synced(cmd_list.get());
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
