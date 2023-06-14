#version 450

layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec2 frag_texcoord;

layout(location = 0) out vec4 out_color;

struct SceneUBO {
  vec3 camera_pos;
  mat4 view;
  mat4 projection;
  vec3 ambient_light;
  int point_light_count;
};

layout(binding = 0) uniform UBORenderable {
  vec3 color;
  int use_per_vertex_color;
  int two_sided;
  float has_attribute;
}
ubo_renderable;

layout(binding = 1) uniform UBOScene {
  SceneUBO scene;
  float window_width;
  float window_height;
  float tan_half_fov;
  float aspect_ratio;
}
ubo_scene;

struct PointLight {
  vec3 pos;
  vec3 color;
};

layout(binding = 2, std430) buffer SSBO {
  PointLight point_lights[];
}
ssbo;

layout(location = 3) in vec4 selected_color;

vec3 lambertian() {
  vec3 ambient = ubo_scene.scene.ambient_light * selected_color.rgb;
  vec3 result = ambient;

  for (int i = 0; i < ubo_scene.scene.point_light_count; ++i) {
    vec3 light_color = ssbo.point_lights[i].color;

    vec3 light_dir = normalize(ssbo.point_lights[i].pos - frag_pos);
    vec3 normal = normalize(frag_normal);
    float factor = 0.0;
    if(ubo_renderable.two_sided != 0){
      factor = abs(dot(light_dir, normal));
    }
    else{
      factor = max(dot(light_dir, normal), 0);
    }
    vec3 diffuse = factor * selected_color.rgb * light_color;
    result += diffuse;
  }

  return result;
}

void main() {
  out_color = vec4(lambertian(), selected_color.a);
}
