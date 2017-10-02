/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "taichi/dynamics/fluid2d/apic.h"

TC_NAMESPACE_BEGIN

APICLiquid::APICLiquid() {}

void APICLiquid::initialize_solver(const Config &config) {
  FLIPLiquid::initialize_solver(config);
  FLIP_alpha = 0.0_f;
  padding = config.get("padding", 0.501f);
  advection_order = config.get("advection_order", 1);
  if (advection_order == 2) {
    printf("Warning: using second order advection can be unstable for APIC!\n");
  }
  apic_blend = config.get("apic_blend", 1.0_f);
  printf("initialized\n");
}

void APICLiquid::rasterize() {
  rasterize_component<Particle::get_affine_velocity<0>>(u, u_count);
  rasterize_component<Particle::get_affine_velocity<1>>(v, v_count);
}

void APICLiquid::sample_c() {
  for (auto &p : particles) {
    p.c[0] = apic_blend * sample_c(p.position, u);
    p.c[1] = apic_blend * sample_c(p.position, v);
  }
}

Vector2 APICLiquid::sample_c(Vector2 &pos, Array<real> &val) {
  const int extent = (1 + 1) / 2;
  Vector2 c(0);
  for (auto &ind : val.get_rasterization_region(pos, extent)) {
    if (!val.inside(ind))
      continue;
    Vector2 grad = grad_kernel(ind.get_pos() - pos);
    c += grad * val[ind];
  }
  return c;
}

void APICLiquid::substep(real delta_t) {
  Time::Timer _("substep");
  apply_external_forces(delta_t);
  mark_cells();
  rasterize();
  if (t == 0.0_f)
    compute_liquid_levelset();
  else {
    advect_liquid_levelset(delta_t);
  }
  simple_extrapolate();
  TIME(project(delta_t));
  simple_extrapolate();
  apply_boundary_condition();
  sample_c();
  advect(delta_t);
  t += delta_t;
}

TC_IMPLEMENTATION(Fluid, APICLiquid, "apic_liquid");

TC_NAMESPACE_END
