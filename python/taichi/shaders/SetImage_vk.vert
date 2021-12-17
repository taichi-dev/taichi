#version 450

// layout(binding = 0) uniform UniformBufferObject {} ubo;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec2 frag_texcoord;

void main() {
  gl_Position = vec4(in_position.xy, 0.0, 1.0);
  frag_texcoord = in_texcoord;
}
