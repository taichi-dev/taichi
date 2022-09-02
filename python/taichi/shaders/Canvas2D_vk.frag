#version 450
precision mediump float;

layout(location = 0) in vec2 v_position;
layout(location = 1) in vec2 v_texcoord;
layout(location = 2) in vec3 v_color;

layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform Ubo {
  vec4 color;
  vec4 wh_invwh;
  float radius;
} ubo;

void main() {
  float radius = ubo.radius;
  vec2 invwh = ubo.wh_invwh.zw;

  // Note that `v_position` is interpolated if the primitive is a line or a
  // triangle so `gl_FragCoord` always touches `v_position`.
  float r = length(gl_FragCoord.xy - v_position);
  if (r >= radius) {
    discard;
  }

  out_color = vec4(v_color, 1);
}
