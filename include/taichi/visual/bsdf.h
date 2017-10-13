/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>
#include <taichi/visual/scene.h>

TC_NAMESPACE_BEGIN

class SurfaceMaterial;
enum class SurfaceScatteringFlags;
class Scene;
struct IntersectionInfo;

// Generalized BSDF (supposed to include photon and importon source)
class BSDF {
 protected:
  SurfaceMaterial *material;
  Matrix3 world_to_local;  // shaded normal
  Matrix3 local_to_world;  // shaded normal
  Vector3 geometry_normal;
  bool front;
  Vector2 uv;

 public:
  BSDF() {
    material = nullptr;
  }

  BSDF(std::shared_ptr<Scene> const &scene, const IntersectionInfo &inter);

  BSDF(std::shared_ptr<Scene> const &scene,
       int triangle_id);  // initialize for light triangle

  real cos_theta(const Vector3 &out) {
    return abs((world_to_local * out).z);
  }

  void sample(const Vector3 &in_dir,
              real u,
              real v,
              Vector3 &out_dir,
              Vector3 &f,
              real &pdf,
              SurfaceEvent &event) const;

  real probability_density(const Vector3 &in, const Vector3 &out) const;

  Vector3 get_geometry_normal() {
    return geometry_normal;
  }

  Vector3 evaluate(const Vector3 &in, const Vector3 &out) const;

  bool is_delta() const;

  bool is_emissive() const;

  bool is_index_matched() const {
    return material->is_index_matched();
  }

  bool is_entering(const Vector3 &in_dir) const {
    return bool((!front) ^ (dot(geometry_normal, in_dir) > 0));
  }

  bool is_leaving(const Vector3 &in_dir) const {
    return !is_entering(in_dir);
  }

  VolumeMaterial const *get_internal_material() const {
    return material->get_internal_material();
  }

  std::string get_name() {
    return "TBD";
  };
};

TC_NAMESPACE_END
