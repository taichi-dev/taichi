#version 460

layout(location = 0) in vec2 v_position;
layout(location = 1) in vec3 v_color;
layout(location = 2) in vec2 v_texcoord;

layout(location = 0) out vec3 color;
layout(location = 1) out vec2 frag_texcoord;

layout(binding = 3) uniform UBO {
  float scale;
}
ubo;

void main() {
  gl_Position = vec4(v_position * ubo.scale, 0.0, 1.0);
  color = v_color;
  frag_texcoord = v_texcoord;
}
