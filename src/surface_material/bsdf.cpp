/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/bsdf.h>
#include <taichi/visual/scene.h>
#include <taichi/visual/surface_material.h>

TC_NAMESPACE_BEGIN

BSDF::BSDF(std::shared_ptr<Scene> const &scene, const IntersectionInfo &inter) {
  world_to_local = Matrix3(inter.to_local);
  local_to_world = Matrix3(inter.to_world);
  geometry_normal = inter.geometry_normal;
  if (inter.material == nullptr)
    material =
        scene->get_mesh_from_triangle_id(inter.triangle_id)->material.get();
  else
    material = inter.material;
  uv = inter.uv;
  front = inter.front;
}

BSDF::BSDF(std::shared_ptr<Scene> const &scene,
           int triangle_id) {  // initialize for light triangle
  Triangle t = scene->get_triangle(triangle_id);
  Vector3 u = normalized(t.v[1] - t.v[0]);
  float sgn = 1;
  Vector3 v = cross(sgn * t.normal, u);
  local_to_world = Matrix3(u, v, t.normal);
  world_to_local = transposed(local_to_world);
  geometry_normal = t.normal;
  material = scene->get_mesh_from_triangle_id(triangle_id)->material.get();
  uv = Vector2(0.5f);
}

void BSDF::sample(const Vector3 &in_dir,
                  real u,
                  real v,
                  Vector3 &out_dir,
                  Vector3 &f,
                  real &pdf,
                  SurfaceEvent &event) const {
  const Vector3 in_dir_local = world_to_local * in_dir;
  Vector3 out_dir_local;
  material->sample(in_dir_local, u, v, out_dir_local, f, pdf, event, uv);
  out_dir = local_to_world * out_dir_local;
}

real BSDF::probability_density(const Vector3 &in, const Vector3 &out) const {
  real pdf = material->probability_density(world_to_local * in,
                                           world_to_local * out, uv);
  assert_info(pdf >= 0, "PDF should be non-negative: " + std::to_string(pdf));
  return pdf;
}

Vector3 BSDF::evaluate(const Vector3 &in, const Vector3 &out) const {
  if (dot(geometry_normal, out) * (world_to_local * out).z <= 0.0f) {
    // for shaded/interpolated normal consistency
    return Vector3(0.0f);
  }
  Vector3 output =
      material->evaluate_bsdf(world_to_local * in, world_to_local * out, uv);
  assert_info(output[0] >= 0 && output[1] >= 0 && output[2] >= 0,
              "BSDF should be non-negative.");
  return output;
}

bool BSDF::is_delta() const {
  assert_info(material != nullptr, "material is empty!");
  return material->is_delta();
}

bool BSDF::is_emissive() const {
  assert_info(material != nullptr, "material is empty!");
  return material->is_emissive();
}

TC_NAMESPACE_END
