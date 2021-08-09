#pragma once

#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

enum class ProjectionMode : int { Perspective = 0, Orthogonal = 1 };

struct Camera {
  glm::vec3 position;
  glm::vec3 lookat;
  glm::vec3 up;
  ProjectionMode projection_mode = ProjectionMode::Perspective;

  float fov{45};
  float left{-1}, right{1}, top{-1}, bottom{1}, z_near{0.1}, z_far{1000};

  glm::mat4 get_view_matrix() {
    return glm::lookAt(position, lookat, up);
  }
  glm::mat4 get_projection_matrix(float aspect_ratio) {
    if (projection_mode == ProjectionMode::Perspective) {
      return glm::perspective(fov, aspect_ratio, 0.1f, 1000.f);
    } else if (projection_mode == ProjectionMode::Orthogonal) {
      return glm::ortho(left, right, top, bottom, z_near, z_far);
    } else {
      throw std::runtime_error("invalid camera projection mode");
    }
  }
};

TI_UI_NAMESPACE_END
