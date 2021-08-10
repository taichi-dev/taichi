#pragma once

#include <vector>
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/ui/common/camera.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct PointLight {
  glm::vec3 pos;
  glm::vec3 color;
};

struct MeshInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
};

struct ParticlesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float radius;
};

class SceneBase {
 public:
  static constexpr int kMaxPointLights = 16;

  void set_camera(const Camera &camera) {
    camera_ = camera;
  }

  void mesh(const MeshInfo &info) {
    mesh_infos_.push_back(info);
  }
  void particles(const ParticlesInfo &info) {
    particles_infos_.push_back(info);
  }
  void point_light(glm::vec3 pos, glm::vec3 color) {
    if (point_lights_.size() >= kMaxPointLights) {
      throw std::runtime_error("point light count exceeds kMaxPointLights");
    }
    point_lights_.push_back({pos, color});
  }
  void ambient_light(glm::vec3 color) {
    ambient_light_color_ = color;
  }
  virtual ~SceneBase() = default;

 protected:
  Camera camera_;
  glm::vec3 ambient_light_color_ = glm::vec3(0.1, 0.1, 0.1);
  std::vector<PointLight> point_lights_;
  std::vector<MeshInfo> mesh_infos_;
  std::vector<ParticlesInfo> particles_infos_;
};

TI_UI_NAMESPACE_END
