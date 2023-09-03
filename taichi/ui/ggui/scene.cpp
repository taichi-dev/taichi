#include "scene.h"
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

namespace vulkan {

using namespace taichi::lang;
void Scene::set_camera(const Camera &camera) {
  camera_ = camera;
}

void Scene::lines(const SceneLinesInfo &info) {
  scene_lines_infos_.push_back(info);
  scene_lines_infos_.back().object_id = next_object_id_++;
}
void Scene::mesh(const MeshInfo &info) {
  mesh_infos_.push_back(info);
  mesh_infos_.back().object_id = next_object_id_++;
}
void Scene::particles(const ParticlesInfo &info) {
  particles_infos_.push_back(info);
  particles_infos_.back().object_id = next_object_id_++;
}
void Scene::point_light(glm::vec3 pos, glm::vec3 color) {
  point_lights_.push_back({glm::vec4(pos, 1.0), glm::vec4(color, 1.0)});
}
void Scene::ambient_light(glm::vec3 color) {
  ambient_light_color_ = color;
}

}  // namespace vulkan

}  // namespace taichi::ui
