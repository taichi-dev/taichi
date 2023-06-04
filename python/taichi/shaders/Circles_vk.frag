#version 450

layout(binding = 0) uniform UBO {
  vec3 color;
  int use_per_vertex_color;
  int use_per_vertex_radius;
  float radius;
  float window_width;
  float window_height;
}
ubo;

layout(location = 1) in vec3 selected_color;
layout(location = 2) in vec2 pos_2d;

layout(location = 0) out vec4 out_color;

void main() {
  float dist_sq = dot(pos_2d, pos_2d);
  float alpha = 1.0 - step(1.0, dist_sq);

  /*
  if (length(pos_2d) >= 1.0) {
    discard;
  }
  */

  out_color = vec4(selected_color, alpha);
}
