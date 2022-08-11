#pragma once

#include "taichi/program/field_info.h"
#include "taichi/ui/common/scene_base.h"
#include "taichi/ui/common/renderable_info.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct SetImageInfo {
  FieldInfo img;
};

struct TrianglesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
};

struct CirclesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float radius{0};
};

struct LinesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float width{0};
};

class CanvasBase {
 public:
  virtual void set_background_color(const glm::vec3 &color) = 0;
  virtual void set_image(const SetImageInfo &info) = 0;
  virtual void triangles(const TrianglesInfo &info) = 0;
  virtual void circles(const CirclesInfo &info) = 0;
  virtual void lines(const LinesInfo &info) = 0;
  virtual void scene(SceneBase *scene) = 0;
  virtual ~CanvasBase() = default;
};

TI_UI_NAMESPACE_END
