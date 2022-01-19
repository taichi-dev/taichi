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
#include "taichi/ui/backends/vulkan/vertex.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/swap_chain.h"
#include "taichi/ui/backends/vulkan/renderable.h"
#include "taichi/ui/common/canvas_base.h"

#include "taichi/ui/backends/vulkan/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#include "taichi/ui/backends/vulkan/renderables/set_image.h"
#include "taichi/ui/backends/vulkan/renderables/triangles.h"
#include "taichi/ui/backends/vulkan/renderables/mesh.h"
#include "taichi/ui/backends/vulkan/renderables/particles.h"
#include "taichi/ui/backends/vulkan/renderables/circles.h"
#include "taichi/ui/backends/vulkan/renderables/lines.h"

namespace taichi {
namespace lang {
class Program;
}  // namespace lang
}  // namespace taichi

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Canvas final : public CanvasBase {
 public:
  Canvas(lang::Program *prog, Renderer *renderer);

  virtual void set_background_color(const glm::vec3 &color) override;

  virtual void set_image(const SetImageInfo &info) override;

  virtual void triangles(const TrianglesInfo &info) override;

  virtual void circles(const CirclesInfo &info) override;

  virtual void lines(const LinesInfo &info) override;

  virtual void scene(SceneBase *scene_base) override;

 private:
  lang::Program *prog_;
  Renderer *renderer_;
};

}  // namespace vulkan

TI_UI_NAMESPACE_END
