#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

void main() {
  out_color = texture(texSampler, frag_texcoord);
  // out_color = vec4(frag_texcoord.xy,0,1);
}
