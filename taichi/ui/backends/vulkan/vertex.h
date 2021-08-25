#pragma once

namespace taichi {
namespace ui {

struct Vertex {
  struct vec3 {
    float x;
    float y;
    float z;
  };
  struct vec2 {
    float x;
    float y;
  };
  vec3 pos;
  vec3 normal;
  vec2 texCoord;
  vec3 color;
};

}  // namespace ui
}  // namespace taichi
