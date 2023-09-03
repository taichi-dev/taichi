#pragma once

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include <memory>

#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/vertex.h"
#include "taichi/ui/ggui/scene.h"
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/renderable.h"
#include "taichi/ui/common/canvas_base.h"

#include "gui.h"
#include "gui_metal.h"

#include "renderables/set_image.h"
#include "renderables/triangles.h"
#include "renderables/mesh.h"
#include "renderables/particles.h"
#include "renderables/circles.h"
#include "renderables/lines.h"
#include "renderables/scene_lines.h"

#ifdef TI_WITH_METAL
#include "nswindow_adapter.h"
#endif

namespace taichi::lang {
class Program;
}  // namespace taichi::lang

namespace taichi::ui {

namespace vulkan {

class TI_DLL_EXPORT Renderer {
 public:
  void init(lang::Program *prog, TaichiWindow *window, const AppConfig &config);
  ~Renderer();

  void prepare_for_next_frame();

  void set_background_color(const glm::vec3 &color);

  void set_image(const SetImageInfo &info);

  void set_image(taichi::lang::Texture *tex);

  void triangles(const TrianglesInfo &info);

  void circles(const CirclesInfo &info);

  void lines(const LinesInfo &info);

  void mesh(const MeshInfo &info);

  void particles(const ParticlesInfo &info);

  void scene_lines(const SceneLinesInfo &info);

  void scene(SceneBase *scene);

  void scene_v2(SceneBase *scene);

  void draw_frame(GuiBase *gui);

  const AppContext &app_context() const;
  AppContext &app_context();
  const SwapChain &swap_chain() const;
  SwapChain &swap_chain();

  taichi::lang::StreamSemaphore get_render_complete_semaphore();

 private:
  void resize_lights_ssbo(int new_ssbo_size);
  void update_scene_data(SceneBase *scene);
  void init_scene_ubo();
  glm::vec3 background_color_ = glm::vec3(0.f, 0.f, 0.f);

  AppContext app_context_;
  SwapChain swap_chain_;

  std::vector<std::unique_ptr<Renderable>> renderables_;
  std::vector<Renderable *> render_queue_;

  DeviceAllocationUnique lights_ssbo_{nullptr};
  unsigned long long lights_ssbo_size{0};
  DeviceAllocationUnique scene_ubo_{nullptr};

  taichi::lang::StreamSemaphore render_complete_semaphore_{nullptr};

  template <typename T>
  T *get_renderable_of_type(VertexAttributes vbo_attrs);
};

}  // namespace vulkan

}  // namespace taichi::ui
