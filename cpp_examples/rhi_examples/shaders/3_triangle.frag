#version 460

layout(location = 0) in vec3 color;
layout(location = 1) in vec2 frag_texcoord;

layout(location = 0) out vec4 frag_output;

layout(binding = 5) uniform sampler2D texSampler;

void main() {
  frag_output = vec4(color * texture(texSampler, frag_texcoord).r, 1.0);
}
