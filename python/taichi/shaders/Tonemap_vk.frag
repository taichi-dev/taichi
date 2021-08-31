#version 450

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D hdr_buffer;
layout(set = 0, binding = 1) uniform sampler2D depth_buffer;

vec3 ACESFilm(vec3 x)
{
  float a = 2.51f;
  float b = 0.03f;
  float c = 2.43f;
  float d = 0.59f;
  float e = 0.14f;
  return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3(0.0), vec3(1.0));
}

#include "color.glslinc"

void main() {
  ivec2 iuv = ivec2(gl_FragCoord.st);

  float depth = texelFetch(depth_buffer, iuv, 0).r;

  if (depth > 0.0)
  {
    const float exposure = 1.0f;

    vec3 hdr_color = texelFetch(hdr_buffer, iuv, 0).rgb;
    hdr_color *= exposure;
    hdr_color = ACESFilm(hdr_color);

    out_color = vec4(toGamma(hdr_color), 1.0);
  }
  else
  {
    discard;
  }
}