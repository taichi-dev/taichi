#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec2 frag_texcoord;
layout(location = 1) out vec3 selected_color;

layout(binding = 0) uniform UniformBufferObject {
  vec3 color;
  int use_per_vertex_color;
}
ubo;

void main() {
  float x = in_position.x * 2.0 - 1.0;
  float y = -(in_position.y * 2.0 - 1.0);

  gl_Position = vec4(x, y, 0.0, 1.0);
  frag_texcoord = in_texcoord;

  if (ubo.use_per_vertex_color == 0) {
    selected_color = ubo.color;
  } else {
    selected_color = in_color.rgb;
  }
}
