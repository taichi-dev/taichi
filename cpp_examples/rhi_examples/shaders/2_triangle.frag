#version 460

layout(location = 0) in vec3 color;

layout(location = 0) out vec4 frag_output;

void main() {
  frag_output = vec4(color, 1.0);
}
