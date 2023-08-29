#include "canvas.h"
#include "taichi/ui/utils/utils.h"
#include "taichi/ui/ggui/sceneV2.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;

Canvas::Canvas(Renderer *renderer) : renderer_(renderer) {
}

void Canvas::set_background_color(const glm::vec3 &color) {
  renderer_->set_background_color(color);
}

void Canvas::set_image(const SetImageInfo &info) {
  renderer_->set_image(info);
}

void Canvas::set_image(Texture *tex) {
  renderer_->set_image(tex);
}

void Canvas::triangles(const TrianglesInfo &info) {
  renderer_->triangles(info);
}

void Canvas::lines(const LinesInfo &info) {
  renderer_->lines(info);
}

void Canvas::circles(const CirclesInfo &info) {
  renderer_->circles(info);
}

void Canvas::scene(SceneBase *scene_base) {
  if (SceneV2 *scene = dynamic_cast<SceneV2 *>(scene_base)) {
    renderer_->scene_v2(scene);
  } else {
    renderer_->scene(scene_base);
  }
}

}  // namespace vulkan

}  // namespace taichi::ui
