#pragma once
#include "taichi/ui/common/scene_base.h"
#include "taichi/ui/ggui/renderer.h"

namespace taichi::ui {

namespace vulkan {

class TI_DLL_EXPORT SceneV2 final : public SceneBase {
 public:
  friend class Renderer;  // later, Renderer wont need to be friend
  friend class Particles;
  friend class Mesh;
  friend class SceneLines;

  explicit SceneV2(Renderer *renderer);

  void set_camera(const Camera &camera) override;
  void lines(const SceneLinesInfo &info) override;
  void mesh(const MeshInfo &info) override;
  void particles(const ParticlesInfo &info) override;
  void point_light(glm::vec3 pos, glm::vec3 color) override;
  void ambient_light(glm::vec3 color) override;

 private:
  Renderer *renderer_;
};

}  // namespace vulkan

}  // namespace taichi::ui
