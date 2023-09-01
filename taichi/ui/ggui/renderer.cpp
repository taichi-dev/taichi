#include "renderer.h"

#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
using namespace taichi::lang::vulkan;
#ifdef TI_WITH_METAL
using namespace taichi::lang::metal;
#endif
void Renderer::init(Program *prog,
                    TaichiWindow *window,
                    const AppConfig &config) {
  switch (config.ggui_arch) {
    case Arch::vulkan:
      app_context_.init_with_vulkan(prog, window, config);
      break;
    case Arch::metal:
      app_context_.init_with_metal(prog, window, config);
      break;
    default:
      throw std::runtime_error("Incorrect arch for GGUI");
  }

  swap_chain_.init(&app_context_);

#ifdef TI_WITH_METAL
  if (config.ggui_arch == Arch::metal) {
    MetalSurface *mtl_surf =
        dynamic_cast<MetalSurface *>(&(swap_chain_.surface()));

    NSWindowAdapter nswin_adapter;
    nswin_adapter.set_content_view(window, mtl_surf);
  }
#endif
}

template <typename T>
T *Renderer::get_renderable_of_type(VertexAttributes vbo_attrs) {
  std::unique_ptr<T> r = std::make_unique<T>(&app_context_, vbo_attrs);
  T *ret = r.get();
  renderables_.push_back(std::move(r));

  return ret;
}

void Renderer::set_background_color(const glm::vec3 &color) {
  background_color_ = color;
}

void Renderer::set_image(const SetImageInfo &info) {
  SetImage *s = get_renderable_of_type<SetImage>(VboHelpers::all());
  s->update_data(info);
  render_queue_.push_back(s);
}

void Renderer::set_image(Texture *tex) {
  SetImage *s = get_renderable_of_type<SetImage>(VboHelpers::all());
  s->update_data(tex);
  render_queue_.push_back(s);
}

void Renderer::triangles(const TrianglesInfo &info) {
  Triangles *triangles =
      get_renderable_of_type<Triangles>(info.renderable_info.vbo_attrs);
  triangles->update_data(info);
  render_queue_.push_back(triangles);
}

void Renderer::lines(const LinesInfo &info) {
  Lines *lines = get_renderable_of_type<Lines>(info.renderable_info.vbo_attrs);
  lines->update_data(info);
  render_queue_.push_back(lines);
}

void Renderer::circles(const CirclesInfo &info) {
  Circles *circles =
      get_renderable_of_type<Circles>(info.renderable_info.vbo_attrs);
  circles->update_data(info);
  render_queue_.push_back(circles);
}

void Renderer::scene_lines(const SceneLinesInfo &info) {
  SceneLines *scene_lines =
      get_renderable_of_type<SceneLines>(info.renderable_info.vbo_attrs);
  scene_lines->update_data(info);
  render_queue_.push_back(scene_lines);
}

void Renderer::mesh(const MeshInfo &info) {
  Mesh *mesh = get_renderable_of_type<Mesh>(info.renderable_info.vbo_attrs);
  mesh->update_data(info);
  render_queue_.push_back(mesh);
}

void Renderer::particles(const ParticlesInfo &info) {
  Particles *particles =
      get_renderable_of_type<Particles>(info.renderable_info.vbo_attrs);
  particles->update_data(info);
  render_queue_.push_back(particles);
}

void Renderer::resize_lights_ssbo(int new_ssbo_size) {
  if (lights_ssbo_ != nullptr && new_ssbo_size == lights_ssbo_size) {
    return;
  }
  lights_ssbo_.reset();
  lights_ssbo_size = new_ssbo_size;
  if (lights_ssbo_size) {
    auto [buf, res] = app_context_.device().allocate_memory_unique(
        {lights_ssbo_size, /*host_write=*/true, /*host_read=*/false,
         /*export_sharing=*/false, AllocUsage::Storage});
    TI_ASSERT(res == RhiResult::success);
    lights_ssbo_ = std::move(buf);
  }
}

void Renderer::init_scene_ubo() {
  scene_ubo_.reset();
  auto [buf, res] = app_context_.device().allocate_memory_unique(
      {sizeof(SceneBase::UBOScene), /*host_write=*/true, /*host_read=*/false,
       /*export_sharing=*/false, AllocUsage::Uniform});
  TI_ASSERT(res == RhiResult::success);
  scene_ubo_ = std::move(buf);
}

void Renderer::update_scene_data(SceneBase *scene) {
  // Update SSBO
  {
    size_t new_ssbo_size = scene->point_lights_.size() * sizeof(PointLight);
    resize_lights_ssbo(new_ssbo_size);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_.device().map(lights_ssbo_->get_ptr(), &mapped));
    memcpy(mapped, scene->point_lights_.data(), new_ssbo_size);
    app_context_.device().unmap(*lights_ssbo_);
  }

  // Update UBO
  {
    init_scene_ubo();

    SceneBase::UBOScene ubo;
    ubo.scene = scene->current_scene_data_;
    ubo.window_width = app_context_.config.width;
    ubo.window_height = app_context_.config.height;
    ubo.tan_half_fov = tanf(glm::radians(scene->camera_.fov) / 2);
    ubo.aspect_ratio =
        float(app_context_.config.width) / float(app_context_.config.height);

    void *mapped{nullptr};
    RHI_VERIFY(app_context_.device().map(scene_ubo_->get_ptr(0), &mapped));
    memcpy(mapped, &ubo, sizeof(ubo));
    app_context_.device().unmap(*scene_ubo_);
  }
}

void Renderer::scene_v2(SceneBase *scene) {
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.width() / (float)swap_chain_.height();
  scene->update_ubo(aspect_ratio);
  update_scene_data(scene);

  for (auto renderable_ : render_queue_) {
    if (renderable_->is_3d_renderable) {
      renderable_->update_scene_data(lights_ssbo_->get_ptr(0),
                                     scene_ubo_->get_ptr(0));
    }
  }

  scene->point_lights_.clear();
}

void Renderer::scene(SceneBase *scene) {
  if (scene->point_lights_.size() == 0) {
    TI_WARN("warning, there are no light sources in the scene.\n");
  }
  float aspect_ratio = swap_chain_.width() / (float)swap_chain_.height();
  scene->update_ubo(aspect_ratio);
  update_scene_data(scene);

  int object_count = scene->mesh_infos_.size() +
                     scene->particles_infos_.size() +
                     scene->scene_lines_infos_.size();
  int mesh_id = 0;
  int particles_id = 0;
  int scene_lines_id = 0;
  for (int i = 0; i < object_count; ++i) {
    if (mesh_id < scene->mesh_infos_.size() &&
        scene->mesh_infos_[mesh_id].object_id == i) {
      mesh(scene->mesh_infos_[mesh_id]);
      ++mesh_id;
    }
    if (particles_id < scene->particles_infos_.size() &&
        scene->particles_infos_[particles_id].object_id == i) {
      particles(scene->particles_infos_[particles_id]);
      ++particles_id;
    }
    // Scene Lines
    if (scene_lines_id < scene->scene_lines_infos_.size() &&
        scene->scene_lines_infos_[scene_lines_id].object_id == i) {
      scene_lines(scene->scene_lines_infos_[scene_lines_id]);
      ++scene_lines_id;
    }
  }
  scene->next_object_id_ = 0;
  scene->mesh_infos_.clear();
  scene->particles_infos_.clear();
  scene->scene_lines_infos_.clear();
  scene->point_lights_.clear();

  for (auto renderable_ : render_queue_) {
    if (renderable_->is_3d_renderable) {
      renderable_->update_scene_data(lights_ssbo_->get_ptr(0),
                                     scene_ubo_->get_ptr(0));
    }
  }
}

Renderer::~Renderer() {
}

void Renderer::prepare_for_next_frame() {
}

void Renderer::draw_frame(GuiBase *gui_base) {
  auto stream = app_context_.device().get_graphics_stream();
  auto [cmd_list, res] = stream->new_command_list_unique();
  assert(res == RhiResult::success && "Failed to allocate command list");

  bool color_clear = true;
  std::vector<float> clear_colors = {background_color_[0], background_color_[1],
                                     background_color_[2], 1};
  auto semaphore = swap_chain_.surface().acquire_next_image();
  auto image = swap_chain_.surface().get_target_image();
  cmd_list->image_transition(image, ImageLayout::undefined,
                             ImageLayout::color_attachment);
  auto depth_image = swap_chain_.depth_allocation();

  for (auto renderable : render_queue_) {
    renderable->record_prepass_this_frame_commands(cmd_list.get());
  }

  cmd_list->begin_renderpass(
      /*x0=*/0, /*y0=*/0, /*x1=*/swap_chain_.width(),
      /*y1=*/swap_chain_.height(), /*num_color_attachments=*/1, &image,
      &color_clear, &clear_colors, &depth_image,
      /*depth_clear=*/true);

  for (auto renderable : render_queue_) {
    renderable->record_this_frame_commands(cmd_list.get());
  }

  if (app_context_.config.ggui_arch == Arch::vulkan) {
    Gui *gui = static_cast<Gui *>(gui_base);
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
  }
#ifdef TI_WITH_METAL
  else if (app_context_.config.ggui_arch == Arch::metal) {
    GuiMetal *gui = static_cast<GuiMetal *>(gui_base);

    auto mtl_cmd_list = static_cast<MetalCommandList *>(cmd_list.get());

    MTLRenderPassDescriptor *pass = mtl_cmd_list->create_render_pass_desc(
        false, mtl_cmd_list->is_renderpass_active());
    mtl_cmd_list->set_renderpass_active();

    gui->init_render_resources(pass);
    gui->draw(cmd_list.get());
  }
#endif
  else {
    TI_NOT_IMPLEMENTED;
  }

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

  render_queue_.clear();
  renderables_.clear();
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

}  // namespace taichi::ui
