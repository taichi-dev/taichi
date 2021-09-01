#version 450

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform UniformBufferObject {
  vec3 color;
  int use_per_vertex_color;
}
ubo;

layout(location = 1) in vec3 selected_color;

void main() {
  out_color = vec4(selected_color, 1);
}
