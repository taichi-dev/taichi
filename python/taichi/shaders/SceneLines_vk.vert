#version 450

layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec3 frag_color;

void main() {
  gl_Position = in_position;
  gl_Position.y = -gl_Position.y;
  frag_color = in_color.rgb;
}
