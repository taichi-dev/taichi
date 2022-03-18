#pragma once

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
  // vec3 normal;
  // vec2 texCoord;
  // vec4 color;
};

}  // namespace ui
}  // namespace taichi
