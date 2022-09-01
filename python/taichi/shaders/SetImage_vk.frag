#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

layout(binding = 1) uniform UBO {
  float x_factor;
  float y_factor;
  int transposed;
}
ubo;

void main() {
  vec2 coord = vec2(ubo.y_factor,ubo.x_factor);
  if (ubo.transposed != 0) {
    coord *= frag_texcoord.yx;
  } else {
    coord *= frag_texcoord.xy;
  }
  out_color = texture(texSampler, coord);
}
