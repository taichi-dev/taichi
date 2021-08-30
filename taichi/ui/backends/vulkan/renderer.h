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
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/backends/vulkan/vertex.h"
#include "taichi/ui/backends/vulkan/scene.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/ui/common/canvas_base.h"

#include "gui.h"
#include <memory>

#include "renderables/set_image.h"
#include "renderables/triangles.h"
#include "renderables/mesh.h"
#include "renderables/particles.h"
#include "renderables/circles.h"
#include "renderables/lines.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Renderer {
 public:
  void init(GLFWwindow *window, const AppConfig &config);
  void cleanup();

  void prepare_for_next_frame();

  void set_background_color(const glm::vec3 &color);

  void set_image(const SetImageInfo &info);

  void triangles(const TrianglesInfo &info);

  void circles(const CirclesInfo &info);

  void lines(const LinesInfo &info);

  void mesh(const MeshInfo &info, Scene *scene);

  void particles(const ParticlesInfo &info, Scene *scene);

  void scene(Scene *scene);

  void draw_frame(Gui *gui);

  const AppContext &app_context() const;
  AppContext &app_context();
  const SwapChain &swap_chain() const;
  SwapChain &swap_chain();

 private:
  glm::vec3 background_color_ = glm::vec3(0.f, 0.f, 0.f);

  std::vector<std::unique_ptr<Renderable>> renderables_;
  int next_renderable_;

  SwapChain swap_chain_;
  AppContext app_context_;

  template <typename T>
  T *get_renderable_of_type();
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
