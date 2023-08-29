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
#include "taichi/ui/ggui/app_context.h"
#include "taichi/ui/ggui/swap_chain.h"
#include "taichi/ui/ggui/renderable.h"
#include "taichi/ui/common/canvas_base.h"

#include "taichi/ui/ggui/gui.h"
#include "taichi/ui/ggui/gui_metal.h"
#include "taichi/ui/ggui/renderer.h"

#include "taichi/ui/ggui/renderables/set_image.h"
#include "taichi/ui/ggui/renderables/triangles.h"
#include "taichi/ui/ggui/renderables/mesh.h"
#include "taichi/ui/ggui/renderables/particles.h"
#include "taichi/ui/ggui/renderables/circles.h"
#include "taichi/ui/ggui/renderables/lines.h"

namespace taichi::ui {

namespace vulkan {

class TI_DLL_EXPORT Canvas final : public CanvasBase {
 public:
  explicit Canvas(Renderer *renderer);

  void set_background_color(const glm::vec3 &color) override;

  void set_image(const SetImageInfo &info) override;

  void set_image(taichi::lang::Texture *tex) override;

  void triangles(const TrianglesInfo &info) override;

  void circles(const CirclesInfo &info) override;

  void lines(const LinesInfo &info) override;

  void scene(SceneBase *scene_base) override;

 private:
  Renderer *renderer_;
};

}  // namespace vulkan

}  // namespace taichi::ui
