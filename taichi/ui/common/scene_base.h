#pragma once

#include <vector>

#include "taichi/program/field_info.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/ui/common/camera.h"
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

struct alignas(16) PointLight {
  glm::vec4 pos;
  glm::vec4 color;
};

struct MeshAttributeInfo {
  FieldInfo mesh_attribute;
  bool has_attribute{false};
};

struct MeshInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  bool two_sided{false};
  int object_id{0};
  int num_instances{1};
  int start_instance{0};
  MeshAttributeInfo mesh_attribute_info;
};

struct ParticlesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float radius{0};
  int object_id{0};
};

struct SceneLinesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float width{0};
  int object_id{0};
};

class SceneBase {
 public:
  virtual void set_camera(const Camera &camera) = 0;
  virtual void lines(const SceneLinesInfo &info) = 0;
  virtual void mesh(const MeshInfo &info) = 0;
  virtual void particles(const ParticlesInfo &info) = 0;
  virtual void point_light(glm::vec3 pos, glm::vec3 color) = 0;
  virtual void ambient_light(glm::vec3 color) = 0;
  virtual ~SceneBase() = default;

 protected:
  Camera camera_;
  glm::vec3 ambient_light_color_ = glm::vec3(0.1, 0.1, 0.1);
  std::vector<PointLight> point_lights_;
  std::vector<SceneLinesInfo> scene_lines_infos_;
  std::vector<MeshInfo> mesh_infos_;
  std::vector<ParticlesInfo> particles_infos_;
  int next_object_id_ = 0;
};

}  // namespace taichi::ui
