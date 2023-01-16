#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

layout(binding = 0) uniform UBO {
  vec3 color;
  int use_per_vertex_color;
  float radius_w;
  float radius_h;
}
ubo;

layout(location = 1) out vec3 selected_color;
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
  float x = in_position.x * 2.0 - 1.0;
  float y = -(in_position.y * 2.0 - 1.0);

  pos_2d = offsets[gl_VertexIndex % 6];

  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_Position.xy += pos_2d * vec2(ubo.radius_w, ubo.radius_h) * 2.0;

  if (ubo.use_per_vertex_color == 0) {
    selected_color = ubo.color;
  } else {
    selected_color = in_color.rgb;
  }
}
