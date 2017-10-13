/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "taichi/dynamics/fluid2d/euler_liquid.h"

TC_NAMESPACE_BEGIN

class FLIPLiquid : public EulerLiquid {
 protected:
  Array<real> u_backup;
  Array<real> v_backup;
  Array<real> u_count;
  Array<real> v_count;
  real FLIP_alpha;
  real padding;
  int advection_order;
  real correction_strength;
  int correction_neighbours;

  void clamp_particle(Particle &p);

  virtual void initialize_solver(const Config &config);

  Vector2 sample_velocity(Vector2 position, Vector2 velocity, real lerp);

  virtual void advect(real delta_t);

  virtual void apply_external_forces(real delta_t);

  virtual void rasterize();

  template <real (*T)(const Particle &, const Vector2 &)>
  void rasterize_component(Array<real> &val, Array<real> &count);

  virtual void backup_velocity_field();

  virtual void substep(real delta_t);

  void reseed();

  void correct_particle_positions(real delta_t, bool clear_c = false);

 public:
  FLIPLiquid() {
  }

  virtual void step(real delta_t);
};

TC_NAMESPACE_END
