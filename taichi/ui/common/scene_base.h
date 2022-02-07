#pragma once

#include <vector>
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/ui/common/camera.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct alignas(16) PointLight {
  glm::vec4 pos;
  glm::vec4 color;
};

struct MeshInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  bool two_sided{false};
  int object_id;
};

struct ParticlesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float radius;
  int object_id;
};

class SceneBase {
 public:
  void set_camera(const Camera &camera) {
    camera_ = camera;
  }

  void mesh(const MeshInfo &info) {
    mesh_infos_.push_back(info);
    mesh_infos_.back().object_id = next_object_id_++;
  }
  void particles(const ParticlesInfo &info) {
    particles_infos_.push_back(info);
    particles_infos_.back().object_id = next_object_id_++;
  }
  void point_light(glm::vec3 pos, glm::vec3 color) {
    point_lights_.push_back({glm::vec4(pos, 1.0), glm::vec4(color, 1.0)});
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
  int next_object_id_ = 0;
};

TI_UI_NAMESPACE_END
