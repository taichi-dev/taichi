#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform UBO {
  float x_factor;
  float y_factor;
  int is_transposed;
} ubo;

void main() {
  vec2 coord = frag_texcoord * vec2(ubo.x_factor,ubo.y_factor);
  out_color = texture(texSampler, ubo.is_transposed != 0 ? coord.yx : coord);
}
