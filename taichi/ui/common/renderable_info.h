#pragma once

#include "taichi/ui/ggui/vertex.h"
#include "taichi/program/field_info.h"
#include "taichi/ui/utils/utils.h"

namespace taichi::ui {

struct RenderableInfo {
  FieldInfo vbo;
  FieldInfo indices;
  bool has_per_vertex_color{false};
  bool has_per_vertex_radius{false};
  VertexAttributes vbo_attrs{VboHelpers::all()};
  bool has_user_customized_draw{false};
  int draw_vertex_count{0};
  int draw_first_vertex{0};
  int draw_index_count{0};
  int draw_first_index{0};
  taichi::lang::PolygonMode display_mode{taichi::lang::PolygonMode::Fill};
};

}  // namespace taichi::ui
