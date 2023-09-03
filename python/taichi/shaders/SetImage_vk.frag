#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform UBO {
  vec2 lower_bound;
  vec2 upper_bound;
  float x_factor;
  float y_factor;
  int is_transposed;
} ubo;

void main() {
  vec2 coord = frag_texcoord * vec2(ubo.x_factor,ubo.y_factor);
  coord = clamp(coord, ubo.lower_bound, ubo.upper_bound);
  out_color = textureLod(texSampler, ubo.is_transposed != 0 ? coord.yx : coord, 0);
}
