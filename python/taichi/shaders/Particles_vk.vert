#version 450

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

layout(binding = 0) uniform UBO {
  SceneUBO scene;
  vec3 color;
  int use_per_vertex_color;
  float radius;
  float window_width;
  float window_height;
  float tan_half_fov;
}
ubo;

layout(location = 0) out vec4 pos_camera_space;
layout(location = 1) out vec4 selected_color;
layout(location = 2) out vec2 pos_2d;

const vec2 offsets[6] = {
  vec2(-1.0f, 1.0f),
  vec2(1.0f, -1.0f),
  vec2(-1.0f, -1.0f),
  vec2(-1.0f, 1.0f),
  vec2(1.0f, 1.0f),
  vec2(1.0f, -1.0f),
};

void main() {
  float distance = length(in_position - ubo.scene.camera_pos);

  float vsize = ubo.radius / (ubo.tan_half_fov * distance);

  pos_camera_space = ubo.scene.view * vec4(in_position, 1.0);
  
  pos_2d = offsets[gl_VertexIndex % 6];
  
  vec4 pos_proj = ubo.scene.projection * pos_camera_space;
  pos_proj = vec4(pos_proj.xyz / pos_proj.w, 1.0);
  pos_proj.xy += pos_2d * vec2(vsize / ubo.window_height * ubo.window_width, vsize);
  
  gl_Position = pos_proj;
  gl_Position.y *= -1.0;

  if (ubo.use_per_vertex_color == 0) {
    selected_color = vec4(ubo.color, 1.0);
  } else {
    selected_color = in_color;
  }
}
