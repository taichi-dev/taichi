#include "renderer.h"

#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;

void Renderer::init(Program *prog,
                    TaichiWindow *window,
                    const AppConfig &config) {
  app_context_.init(prog, window, config);
  swap_chain_.init(&app_context_);
}

template <typename T>
std::unique_ptr<Renderable> get_new_renderable(AppContext *app_context,
                                               VertexAttributes vbo_attrs) {
  return std::unique_ptr<Renderable>{new T(app_context, vbo_attrs)};
}

template <typename T>
T *Renderer::get_renderable_of_type(VertexAttributes vbo_attrs) {
  if (next_renderable_ >= renderables_.size()) {
    renderables_.push_back(get_new_renderable<T>(&app_context_, vbo_attrs));
  } else if (dynamic_cast<T *>(renderables_[next_renderable_].get()) ==
             nullptr) {
    renderables_.insert(renderables_.begin() + next_renderable_,
                        get_new_renderable<T>(&app_context_, vbo_attrs));
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
  SetImage *s = get_renderable_of_type<SetImage>(VboHelpers::all());
  s->update_data(info);
  next_renderable_ += 1;
}

void Renderer::triangles(const TrianglesInfo &info) {
  Triangles *triangles =
      get_renderable_of_type<Triangles>(info.renderable_info.vbo_attrs);
  triangles->update_data(info);
  next_renderable_ += 1;
}

void Renderer::lines(const LinesInfo &info) {
  Lines *lines = get_renderable_of_type<Lines>(info.renderable_info.vbo_attrs);
  lines->update_data(info);
  next_renderable_ += 1;
}

void Renderer::circles(const CirclesInfo &info) {
  Circles *circles =
      get_renderable_of_type<Circles>(info.renderable_info.vbo_attrs);
  circles->update_data(info);
  next_renderable_ += 1;
}

void Renderer::scene_lines(const SceneLinesInfo &info, Scene *scene) {
  SceneLines *scene_lines =
      get_renderable_of_type<SceneLines>(info.renderable_info.vbo_attrs);
  scene_lines->update_data(info, *scene);
  next_renderable_ += 1;
}

void Renderer::mesh(const MeshInfo &info, Scene *scene) {
  Mesh *mesh = get_renderable_of_type<Mesh>(info.renderable_info.vbo_attrs);
  mesh->update_data(info, *scene);
  next_renderable_ += 1;
}

void Renderer::particles(const ParticlesInfo &info, Scene *scene) {
  Particles *particles =
      get_renderable_of_type<Particles>(info.renderable_info.vbo_attrs);
  particles->update_data(info, *scene);
  next_renderable_ += 1;
}

void Renderer::scene(Scene *scene) {
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.width() / (float)swap_chain_.height();
  scene->update_ubo(aspect_ratio);

  int object_count = scene->mesh_infos_.size() +
                     scene->particles_infos_.size() +
                     scene->scene_lines_infos_.size();
  int mesh_id = 0;
  int particles_id = 0;
  int scene_lines_id = 0;
  for (int i = 0; i < object_count; ++i) {
    if (mesh_id < scene->mesh_infos_.size() &&
        scene->mesh_infos_[mesh_id].object_id == i) {
      mesh(scene->mesh_infos_[mesh_id], scene);
      ++mesh_id;
    }
    if (particles_id < scene->particles_infos_.size() &&
        scene->particles_infos_[particles_id].object_id == i) {
      particles(scene->particles_infos_[particles_id], scene);
      ++particles_id;
    }
    // Scene Lines
    if (scene_lines_id < scene->scene_lines_infos_.size() &&
        scene->scene_lines_infos_[scene_lines_id].object_id == i) {
      scene_lines(scene->scene_lines_infos_[scene_lines_id], scene);
      ++scene_lines_id;
    }
  }
  scene->next_object_id_ = 0;
  scene->mesh_infos_.clear();
  scene->particles_infos_.clear();
  scene->scene_lines_infos_.clear();
  scene->point_lights_.clear();
}

Renderer::~Renderer() {
  cleanup();
}

void Renderer::cleanup() {
  render_complete_semaphore_ = nullptr;
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
  auto stream = app_context_.device().get_graphics_stream();
  auto cmd_list = stream->new_command_list();
  bool color_clear = true;
  std::vector<float> clear_colors = {background_color_[0], background_color_[1],
                                     background_color_[2], 1};
  auto semaphore = swap_chain_.surface().acquire_next_image();
  auto image = swap_chain_.surface().get_target_image();
  cmd_list->image_transition(image, ImageLayout::undefined,
                             ImageLayout::color_attachment);
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

  std::vector<StreamSemaphore> wait_semaphores;

  if (app_context_.prog()) {
    auto sema = app_context_.prog()->flush();
    if (sema) {
      wait_semaphores.push_back(sema);
    }
  }

  if (semaphore) {
    wait_semaphores.push_back(semaphore);
  }

  render_complete_semaphore_ = stream->submit(cmd_list.get(), wait_semaphores);
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

taichi::lang::StreamSemaphore Renderer::get_render_complete_semaphore() {
  return std::move(render_complete_semaphore_);
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
