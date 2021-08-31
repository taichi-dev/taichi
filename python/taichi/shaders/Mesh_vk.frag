#version 450

layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

struct SceneUBO {
  vec3 camera_pos;
  mat4 view;
  mat4 projection;
  vec3 ambient_light;
  int point_light_count;
};

layout(binding = 0) uniform UBO {
  SceneUBO scene;
  vec3 color;
  int use_per_vertex_color;
  int two_sided;
}
ubo;

#include "lighting.glslinc"

layout(location = 3) in vec3 selected_color;

void main() {
  vec3 radiance = ubo.scene.ambient_light;
  radiance += get_point_light_radiance(frag_pos, normalize(frag_normal));
  radiance *= selected_color;

  out_color = vec4(radiance, 1.0f);
}
