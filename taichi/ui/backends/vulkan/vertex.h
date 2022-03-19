#pragma once

#include <stddef.h>

#include <type_traits>

namespace taichi {
namespace ui {

struct Vertex {
  struct vec3 {
    float x;
    float y;
    float z;
  };
  struct vec4 {
    float x;
    float y;
    float z;
    float w;
  };
  struct vec2 {
    float x;
    float y;
  };
  vec3 pos;
  vec3 normal;
  vec2 tex_coord;
  vec4 color;
};

enum class VertexAttributes : char {
  kPos = 0b0001,
  kNormal = 0b0010,
  kUv = 0b0100,
  kColor = 0b1000,
};

constexpr inline VertexAttributes operator|(VertexAttributes src,
                                            VertexAttributes a) {
  using UT = std::underlying_type_t<VertexAttributes>;
  return static_cast<VertexAttributes>(UT(src) | UT(a));
}

class VboHelpers {
 public:
  constexpr static VertexAttributes kOrderedAttrs[] = {
      VertexAttributes::kPos,
      VertexAttributes::kNormal,
      VertexAttributes::kUv,
      VertexAttributes::kColor,
  };

  constexpr static VertexAttributes empty() {
    return static_cast<VertexAttributes>(0);
  }
  constexpr static VertexAttributes all() {
    return VertexAttributes::kPos | VertexAttributes::kNormal |
           VertexAttributes::kUv | VertexAttributes::kColor;
  }

  static size_t size(VertexAttributes va);

  static bool has_attr(VertexAttributes src, VertexAttributes attr) {
    using UT = std::underlying_type_t<VertexAttributes>;
    return UT(src) & UT(attr);
  }
};

}  // namespace ui
}  // namespace taichi
