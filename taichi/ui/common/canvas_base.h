#pragma once
#include "taichi/ui/common/field_info.h"
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
  float radius;
};

struct LinesInfo {
  RenderableInfo renderable_info;
  glm::vec3 color;
  float width;
};

struct CanvasBase {
  virtual void set_background_color(const glm::vec3 &color) {
  }
  virtual void set_image(const SetImageInfo &info) {
  }
  virtual void triangles(const TrianglesInfo &info) {
  }
  virtual void circles(const CirclesInfo &info) {
  }
  virtual void lines(const LinesInfo &info) {
  }
  virtual void scene(SceneBase *scene) {
  }
};

TI_UI_NAMESPACE_END
