/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <taichi/common/util.h>
#include "array.h"
#include "math.h"

TC_NAMESPACE_BEGIN

// Rasterized level set
template <int DIM>
class LevelSet : public ArrayND<DIM, real> {
 public:
  using VectorI = VectorND<DIM, int>;
  using Vector = VectorND<DIM, real>;
  using Array = ArrayND<DIM, real>;
  static constexpr real INF = 1e7f;

  real friction = 1.0_f;

  LevelSet() {
    initialize(VectorI(0));
  }

  LevelSet(const VectorI &res, Vector offset = Vector(0.5f), real value = INF) {
    Array::initialize(res, value, offset);
  }

  void initialize(const VectorI &res,
                  Vector offset = Vector(0.5f),
                  real value = INF) {
    Array::initialize(res, value, offset);
  }

  // 2D

  void add_sphere(Vector center, real radius, bool inside_out = false);

  void add_polygon(std::vector<Vector2> polygon, bool inside_out = false);

  real get(const Vector &pos) const;

  Array rasterize(VectorI output_res);

  // 3D

  void add_plane(const Vector &normal, real d);

  void add_cuboid(Vector3 lower_boundry,
                  Vector3 upper_boundry,
                  bool inside_out = true);

  void add_slope(const Vector &center, real radius, real angle);

  void add_cylinder(const Vector &center, real radius, bool inside_out = true);

  void global_increase(real delta);

  Vector get_gradient(const Vector &pos) const;  // Note this is not normalized!

  Vector get_normalized_gradient(const Vector &pos) const;

  static real fraction_inside(real phi_a, real phi_b) {
    if (phi_a < 0 && phi_b < 0)
      return 1;
    if (phi_a < 0 && phi_b >= 0)
      return phi_a / (phi_a - phi_b);
    if (phi_a >= 0 && phi_b < 0)
      return phi_b / (phi_b - phi_a);
    else
      return 0;
  }

  static real fraction_outside(real phi_a, real phi_b) {
    return 1.0_f - fraction_inside(phi_a, phi_b);
  }
};

typedef LevelSet<2> LevelSet2D;
typedef LevelSet<3> LevelSet3D;

template <int DIM>
class DynamicLevelSet {
 public:
  static constexpr int D = DIM;
  using Vector = VectorND<DIM, real>;
  using Vectori = VectorND<DIM, int>;
  real t0, t1;
  std::shared_ptr<LevelSet<DIM>> levelset0, levelset1;

  bool inside(const Vectori pos) const {
    return levelset0->inside(pos.template cast<real>());
  }

  bool inside(const Vector pos) const {
    return levelset0->inside(pos);
  }

  void initialize(real _t0,
                  real _t1,
                  const LevelSet<DIM> &_ls0,
                  const LevelSet<DIM> &_ls1);

  // returns gradient (normalized)
  Vector get_spatial_gradient(const Vector &pos, real t) const;

  real get_temporal_derivative(const Vector &pos, real t) const;

  real sample(const Vector &pos, real t) const;

  ArrayND<DIM, real> rasterize(Vectori res, real t);
};

typedef DynamicLevelSet<2> DynamicLevelSet2D;
typedef DynamicLevelSet<3> DynamicLevelSet3D;

TC_NAMESPACE_END
