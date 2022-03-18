#pragma once

#include "taichi/ui/backends/vulkan/vertex.h"
#include "taichi/ui/common/field_info.h"
#include "taichi/ui/utils/utils.h"

TI_UI_NAMESPACE_BEGIN

struct RenderableInfo {
  FieldInfo vbo;
  FieldInfo indices;
  bool has_per_vertex_color{false};
  VboAttribes vbo_attrs{VboAttribes::kAll};
};

TI_UI_NAMESPACE_END
