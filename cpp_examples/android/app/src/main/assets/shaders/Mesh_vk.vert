#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec3 frag_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec2 frag_texcoord;
layout(location = 3) out vec4 selected_color;

struct SceneUBO {
  vec3 camera_pos;
  mat4 view;
  mat4 projection;
  vec3 ambient_light;
  int point_light_count;
};

struct PointLight {
  vec3 pos;
  vec3 color;
};

layout(binding = 0) uniform UBO {
  SceneUBO scene;
  vec3 color;
  int use_per_vertex_color;
  int two_sided;
}
ubo;

void main() {
  gl_Position = ubo.scene.projection * ubo.scene.view * vec4(in_position, 1.0);
  gl_Position.y *= -1.0;
  frag_texcoord = in_texcoord;
  frag_pos = in_position;
  frag_normal = in_normal;

  if (ubo.use_per_vertex_color == 0) {
    selected_color = vec4(ubo.color, 1.0);
  } else {
    selected_color = in_color;
  }
}
