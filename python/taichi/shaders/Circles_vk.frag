#version 450

layout(binding = 0) uniform UBO {
  vec3 color;
  int use_per_vertex_color;
  float radius;
}
ubo;

layout(location = 1) in vec3 selected_color;

layout(location = 0) out vec4 out_color;

void main() {
  vec2 coord2D;
  coord2D = gl_PointCoord * 2.0 - vec2(1);

  if (length(coord2D) >= 1.0) {
    discard;
  }

  out_color = vec4(selected_color, 1.0);
}
