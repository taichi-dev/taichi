#include "sceneV2.h"
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;

SceneV2::SceneV2(Renderer *renderer) : renderer_(renderer) {
}

void SceneV2::set_camera(const Camera &camera) {
  camera_ = camera;
}

void SceneV2::lines(const SceneLinesInfo &info) {
  renderer_->scene_lines(info);
}
void SceneV2::mesh(const MeshInfo &info) {
  renderer_->mesh(info);
}
void SceneV2::particles(const ParticlesInfo &info) {
  renderer_->particles(info);
}
void SceneV2::point_light(glm::vec3 pos, glm::vec3 color) {
  point_lights_.push_back({glm::vec4(pos, 1.0), glm::vec4(color, 1.0)});
}
void SceneV2::ambient_light(glm::vec3 color) {
  ambient_light_color_ = color;
}

}  // namespace vulkan

}  // namespace taichi::ui
