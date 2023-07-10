#version 450
#extension GL_KHR_vulkan_glsl : enable

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

struct SceneUBO {
  vec3 camera_pos;
  mat4 view;
  mat4 projection;
  vec3 ambient_light;
  int point_light_count;
};

layout(binding = 0) uniform UBORenderable {
  vec3 color;
  int use_per_vertex_color;
  int use_per_vertex_radius;
  float radius;
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
layout(location = 0) out vec4 pos_camera_space;
layout(location = 1) out vec4 selected_color;
layout(location = 2) out vec2 pos_2d;
layout(location = 3) out float selected_radius;

const vec2 offsets[6] = {
  vec2(-1.0f, 1.0f),
  vec2(1.0f, -1.0f),
  vec2(-1.0f, -1.0f),
  vec2(-1.0f, 1.0f),
  vec2(1.0f, 1.0f),
  vec2(1.0f, -1.0f),
};

void main() {
  float distance = length(in_position - ubo_scene.scene.camera_pos);

  if (ubo_renderable.use_per_vertex_radius == 0) {
    selected_radius = ubo_renderable.radius;
  } else {
    selected_radius = in_normal.x;
  }

  float hsize = selected_radius / (ubo_scene.tan_half_fov * distance);

  pos_camera_space = ubo_scene.scene.view * vec4(in_position, 1.0);

  pos_2d = offsets[gl_VertexIndex % 6];

  vec4 pos_proj = ubo_scene.scene.projection * pos_camera_space;
  pos_proj.xy += pos_2d * vec2(hsize, hsize * ubo_scene.window_width / ubo_scene.window_height) * pos_proj.w;

  gl_Position = pos_proj;
  gl_Position.y *= -1.0;

  if (ubo_renderable.use_per_vertex_color == 0) {
    selected_color = vec4(ubo_renderable.color, 1.0);
  } else {
    selected_color = in_color;
  }
}
