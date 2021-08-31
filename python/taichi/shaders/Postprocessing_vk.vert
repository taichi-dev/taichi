#version 450

vec2 positions[] = {
  vec2(-1.0, 4.0),
  vec2(-1.0, -1.0),
  vec2(4.0, -1.0)
};

layout(location = 0) out vec2 uv;

void main() {
  uv = positions[gl_VertexIndex] * 0.5 + 0.5;
  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}