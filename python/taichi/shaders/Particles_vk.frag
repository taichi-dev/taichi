#version 450

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
  int use_per_vertex_radius;
  float radius;
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

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec4 pos_camera_space;
layout(location = 1) in vec4 selected_color;
layout(location = 2) in vec2 pos_2d;
layout(location = 3) in float selected_radius;

float project_z(float view_z) {
  vec4 projected = ubo_scene.scene.projection * vec4(0, 0, view_z, 1);
  return projected.z / projected.w;
}

vec3 to_camera_space(vec3 pos) {
  vec4 temp = ubo_scene.scene.view * vec4(pos, 1.0);
  return temp.xyz / temp.w;
}

// operates in camera space !!
vec3 lambertian(vec3 frag_pos, vec3 frag_normal) {
  vec3 ambient = ubo_scene.scene.ambient_light * selected_color.rgb;
  vec3 result = ambient;

  for (int i = 0; i < ubo_scene.scene.point_light_count; ++i) {
    vec3 light_color = ssbo.point_lights[i].color;

    vec3 light_dir =
        normalize(to_camera_space(ssbo.point_lights[i].pos) - frag_pos);
    vec3 normal = normalize(frag_normal);
    vec3 diffuse =
        max(dot(light_dir, normal), 0.0) * selected_color.rgb * light_color;

    result += diffuse;
  }

  return result;
}

void main() {
  vec2 coord2D = pos_2d;

  if (length(coord2D) >= 1.0) {
    discard;
  }

  float z_in_sphere = sqrt(1 - coord2D.x * coord2D.x - coord2D.y * coord2D.y);
  vec3 coord_in_sphere = vec3(coord2D, z_in_sphere);

  vec3 frag_pos =
      pos_camera_space.xyz / pos_camera_space.w + coord_in_sphere * selected_radius;
  vec3 frag_normal = coord_in_sphere;
  vec3 color = lambertian(frag_pos, frag_normal);
  out_color = vec4(color, selected_color.a);

  float depth =
      (pos_camera_space.z / pos_camera_space.w) + z_in_sphere * selected_radius;

  gl_FragDepth = project_z(depth);
}
