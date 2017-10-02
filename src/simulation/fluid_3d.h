/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory.h>
#include <string>
#include <taichi/visualization/image_buffer.h>
#include <taichi/common/meta.h>
#include <taichi/math/array_3d.h>
#include <taichi/dynamics/poisson_solver.h>
#include <taichi/dynamics/simulation.h>
#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

class Tracker3D {
 public:
  Vector3 position;
  Vector3 color;

  Tracker3D() {}

  Tracker3D(const Vector3 &position, const Vector3 &color)
      : position(position), color(color) {}
};

class Smoke3D : public Simulation3D {
  typedef Array3D<real> Array;

 public:
  Array u, v, w, rho, t, pressure, last_pressure;
  Vector3i res;
  real smoke_alpha, smoke_beta;
  real temperature_decay;
  real pressure_tolerance;
  real density_scaling;
  real tracker_generation;
  real perturbation;
  int super_sampling;
  std::shared_ptr<Texture> generation_tex;
  std::shared_ptr<Texture> initial_velocity_tex;
  std::shared_ptr<Texture> color_tex;
  std::shared_ptr<Texture> temperature_tex;

  bool open_boundary;
  std::vector<Tracker3D> trackers;
  std::shared_ptr<PoissonSolver3D> pressure_solver;
  PoissonSolver3D::BCArray boundary_condition;

  Smoke3D() {}

  void remove_outside_trackers();

  void initialize(const Config &config) override;

  void project();

  void confine_vorticity(real delta_t);

  void advect(real delta_t);

  void move_trackers(real delta_t);

  void step(real delta_t) override;

  virtual void show(Array2D<Vector3> &buffer);

  void advect(Array &attr, real delta_t);

  void apply_boundary_condition();

  static Vector3 sample_velocity(const Array &u,
                                 const Array &v,
                                 const Array &w,
                                 const Vector3 &pos);

  Vector3 sample_velocity(const Vector3 &pos) const;

  std::vector<RenderParticle> get_render_particles() const override;

  void update(const Config &config) override;
};

TC_NAMESPACE_END
