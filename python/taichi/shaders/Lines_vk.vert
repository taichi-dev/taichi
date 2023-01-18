#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in uint in_color_encoded;

layout(location = 0) out vec3 frag_color;

void main() {
  float x = in_position.x * 2.0 - 1.0;
  float y = -(in_position.y * 2.0 - 1.0);

  gl_Position = vec4(x, y, 0.0, 1.0);
  frag_color = unpackUnorm4x8(in_color_encoded).rgb;
}
