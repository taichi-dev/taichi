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

struct MeshAttribute {
  mat4 model;
};

layout(binding = 0) uniform UBORenderable {
  vec3 color;
  int use_per_vertex_color;
  int two_sided;
  float has_attribute;
}
ubo_renderable;

layout(binding = 1) uniform UBOScene {
  SceneUBO scene;
  float window_width;
  float window_height;
  float tan_half_fov;
  float aspect_ratio;
}
ubo_scene;

layout(binding = 3, std430) buffer MeshAttributeBuffer {
  MeshAttribute mesh_attr[];
}
mesh_attr_buffer;

void main() {
  mat4 model_tmp = transpose(mesh_attr_buffer.mesh_attr[gl_InstanceIndex].model);
  // if mesh attributes not given, then use Identity as model matrix
  mat4 model = mat4(1.0) * (1.0 - ubo_renderable.has_attribute) + model_tmp * ubo_renderable.has_attribute;

  gl_Position = ubo_scene.scene.projection * ubo_scene.scene.view * model * vec4(in_position, 1.0);
  gl_Position.y *= -1.0;
  frag_texcoord = in_texcoord;
  frag_pos = in_position;
  frag_normal = in_normal;

  if (ubo_renderable.use_per_vertex_color == 0) {
    selected_color = vec4(ubo_renderable.color, 1.0);
  } else {
    selected_color = in_color;
  }
}
