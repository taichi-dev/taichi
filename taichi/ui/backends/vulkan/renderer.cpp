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
std::unique_ptr<Renderable> get_new_renderable(Renderer *r) {
  return std::unique_ptr<Renderable>{new T(r)};
}

template <typename T>
T *Renderer::get_renderable_of_type() {
  if (next_renderable_ >= renderables_.size()) {
    renderables_.push_back(get_new_renderable<T>(this));
  } else if (dynamic_cast<T *>(renderables_[next_renderable_].get()) ==
             nullptr) {
    renderables_.insert(renderables_.begin() + next_renderable_,
                        get_new_renderable<T>(this));
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

Renderer::~Renderer() {
  for (auto &renderable : renderables_) {
    renderable->cleanup();
  }
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


  std::unique_ptr<CommandList> cmd_list = app_context().vulkan_device().new_command_list({CommandListType::Graphics});
  bool color_clear = true;
  std::vector<float> clear_colors = {background_color_[0],background_color_[1],background_color_[2],1};
  auto image = swap_chain_.surface().get_target_image();
  auto depth_image = swap_chain_.depth_allocation();
  cmd_list->begin_renderpass(0,0,swap_chain_.width(),swap_chain_.height(),1,&image,&color_clear,&clear_colors,&depth_image,true);

    
  for (int i = 0; i < next_renderable_; ++i) {
    renderables_[i]->record_this_frame_commands(cmd_list.get());
  }

  VkRenderPass pass = static_cast<VulkanCommandList*>(cmd_list.get())->current_renderpass();
  
  if(gui->render_pass()==VK_NULL_HANDLE){
    gui->init_render_resources(pass);
  }
  else if(gui->render_pass() != pass){
    gui->cleanup_render_resources();
    gui->init_render_resources(pass);
  }

  gui->draw(cmd_list.get());
  cmd_list->end_renderpass();
  app_context_.vulkan_device().submit_synced(cmd_list.get());

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
