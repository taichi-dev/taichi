// This is the unified vertex shader for all canvas UI draw.
#version 450
precision mediump float;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec2 v_position;
layout(location = 1) out vec2 v_texcoord;
layout(location = 2) out vec3 v_color;

layout(binding = 0) uniform Ubo {
  vec4 color;
  vec4 wh_invwh;
  float radius;
} ubo;

void main() {
  float x = in_position.x * 2.0 - 1.0;
  float y = -(in_position.y * 2.0 - 1.0);
  vec2 wh = ubo.wh_invwh.xy;

  vec3 color = ubo.color.rgb;
  float use_vertex_color = ubo.color.a;
  float radius = ubo.radius;

  v_position = vec2(in_position.x, 1.0 - in_position.y) * wh;
  v_texcoord = in_texcoord;
  gl_Position = vec4(x, y, 0.0, 1.0);
  gl_PointSize = radius * 2.0;

  v_color = mix(color, in_color.rgb, use_vertex_color);
}
