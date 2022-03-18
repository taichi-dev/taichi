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
  vec3 normal;
  // FIXME: tex_coord
  vec2 texCoord;
  vec4 color;
};

enum class VboAttribes {
  kAll,
  kPos,
  kPosNormal,
  kPosNormalUv,
};

size_t sizeof_vbo(VboAttribes va);

}  // namespace ui
}  // namespace taichi
