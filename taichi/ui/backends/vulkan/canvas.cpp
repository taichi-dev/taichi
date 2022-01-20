#include "canvas.h"
#include "taichi/ui/utils/utils.h"

using taichi::lang::Program;

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

using namespace taichi::lang;

Canvas::Canvas(Program *prog, Renderer *renderer)
    : prog_(prog), renderer_(renderer) {
}

void Canvas::set_background_color(const glm::vec3 &color) {
  renderer_->set_background_color(color);
}

void Canvas::set_image(const SetImageInfo &info) {
  renderer_->set_image(prog_, info);
}

void Canvas::triangles(const TrianglesInfo &info) {
  renderer_->triangles(prog_, info);
}

void Canvas::lines(const LinesInfo &info) {
  renderer_->lines(prog_, info);
}

void Canvas::circles(const CirclesInfo &info) {
  renderer_->circles(prog_, info);
}

void Canvas::scene(SceneBase *scene_base) {
  if (Scene *scene = dynamic_cast<Scene *>(scene_base)) {
    renderer_->scene(prog_, scene);
  } else {
    throw std::runtime_error("Scene is not vulkan scene");
  }
}

}  // namespace vulkan

TI_UI_NAMESPACE_END
