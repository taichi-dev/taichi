#pragma once
#include "taichi/ui/common/scene_base.h"

TI_UI_NAMESPACE_BEGIN

namespace vulkan {

class Scene final : public SceneBase {
 public:
  friend class Renderer;
  friend class Particles;
  friend class Mesh;

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

TI_UI_NAMESPACE_END
