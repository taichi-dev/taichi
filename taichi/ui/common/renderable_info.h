#pragma once
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct RenderableInfo {
  FieldInfo vbo;
  FieldInfo indices;
  bool has_per_vertex_color;
  bool has_per_vertex_radius;
};

TI_UI_NAMESPACE_END
