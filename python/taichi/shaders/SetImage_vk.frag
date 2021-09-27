#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform UBO {
  float x_factor;
  float y_factor;
}
ubo;

void main() {
  vec2 coord = frag_texcoord.yx * vec2(ubo.y_factor,ubo.x_factor);
  out_color = texture(texSampler, coord);
  // out_color = vec4(frag_texcoord.xy,0,1);
}
