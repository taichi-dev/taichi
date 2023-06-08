#pragma once
#include "taichi/ui/common/scene_base.h"

namespace taichi::ui {

namespace vulkan {

class Scene final : public SceneBase {
 public:
  friend class Renderer;
  friend class Particles;
  friend class Mesh;
  friend class SceneLines;

  void set_camera(const Camera &camera) override;
  void lines(const SceneLinesInfo &info) override;
  void mesh(const MeshInfo &info) override;
  void particles(const ParticlesInfo &info) override;
  void point_light(glm::vec3 pos, glm::vec3 color) override;
  void ambient_light(glm::vec3 color) override;

 private:
  struct SceneUniformBuffer {
    alignas(16) glm::vec3 camera_pos;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 projection;
    alignas(16) glm::vec3 ambient_light;
    int point_light_count;
  };
  SceneUniformBuffer current_ubo_;

  void update_ubo(float aspect_ratio) {
    current_ubo_.camera_pos = camera_.position;
    current_ubo_.view = camera_.get_view_matrix();
    current_ubo_.projection = camera_.get_projection_matrix(aspect_ratio);
    current_ubo_.point_light_count = point_lights_.size();

    current_ubo_.ambient_light = ambient_light_color_;
  }
};

}  // namespace vulkan

}  // namespace taichi::ui
