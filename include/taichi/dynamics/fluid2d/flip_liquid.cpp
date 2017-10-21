/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "flip_liquid.h"
#include <taichi/nearest_neighbour/point_cloud.h>

TC_NAMESPACE_BEGIN

void FLIPLiquid::clamp_particle(Particle &p) {
  p.position = EulerLiquid::clamp_particle_position(
      p.position);  // Avoid out of bound levelset query
  Vector2 sample_position = p.position;
  real phi = boundary_levelset.sample(sample_position) - padding;
  if (phi < 0) {
    auto grad = boundary_levelset.get_normalized_gradient(sample_position);
    p.position -= phi * grad;
    p.velocity -= dot(grad, p.velocity) * grad;
  }
}

void FLIPLiquid::initialize_solver(const Config &config) {
  EulerLiquid::initialize_solver(config);
  FLIP_alpha = config.get("flip_alpha", 0.97f);
  padding = config.get("padding", 0.001f);
  advection_order = config.get("advection_order", 2);
  correction_strength = config.get("correction_strength", 0.1f);
  correction_neighbours = config.get("correction_neighbours", 5);
  u_backup = Array<real>(u.get_res(), 0.0_f, Vector2(0.0_f, 0.5f));
  v_backup = Array<real>(v.get_res(), 0.0_f, Vector2(0.5f, 0.0_f));
  u_count = Array<real>(u.get_res(), 0.0_f);
  v_count = Array<real>(v.get_res(), 0.0_f);
}

Vector2 FLIPLiquid::sample_velocity(Vector2 position,
                                    Vector2 velocity,
                                    real lerp) {
  return EulerLiquid::sample_velocity(position, u, v) +
         lerp * (velocity -
                 EulerLiquid::sample_velocity(position, u_backup, v_backup));
}

void FLIPLiquid::advect(real delta_t) {
  real lerp = powf(FLIP_alpha, delta_t / 0.01f);
  real max_movement = 0.0_f;
  for (auto &p : particles) {
    if (advection_order == 3) {
      Vector2 velocity_1 = sample_velocity(p.position, p.velocity, lerp);
      Vector2 velocity_2 = sample_velocity(
          (p.position + delta_t * 0.5_f * velocity_1), p.velocity, lerp);
      Vector2 velocity_3 = sample_velocity(
          (p.position + delta_t * 0.75_f * velocity_2), p.velocity, lerp);
      p.velocity = (2.0_f / 9.0_f) * velocity_1 + (3.0_f / 9.0_f) * velocity_2 +
                   (4.0_f / 9.0_f) * velocity_3;
    } else if (advection_order == 2) {
      Vector2 velocity_1 = sample_velocity(p.position, p.velocity, lerp);
      Vector2 velocity_2 =
          sample_velocity(p.position - delta_t * velocity_1, p.velocity, lerp);
      p.velocity = 0.5_f * (velocity_1 + velocity_2);
    } else if (advection_order == 1) {
      p.velocity = sample_velocity(p.position, p.velocity, lerp);
    } else {
      TC_ERROR("advection_order must be in [1, 2, 3].")
    }
    p.move(delta_t * p.velocity);
    max_movement = std::max(max_movement, length(p.velocity * delta_t));
    clamp_particle(p);
  }
}

void FLIPLiquid::apply_external_forces(real delta_t) {
  for (auto &p : particles) {
    p.velocity += delta_t * gravity;
  }
}

void FLIPLiquid::rasterize() {
  rasterize_component<Particle::get_velocity<0>>(u, u_count);
  rasterize_component<Particle::get_velocity<1>>(v, v_count);
}

void FLIPLiquid::step(real delta_t) {
  EulerLiquid::step(delta_t);
  correct_particle_positions(delta_t);
}

void FLIPLiquid::backup_velocity_field() {
  u_backup = u;
  v_backup = v;
}

void FLIPLiquid::substep(real delta_t) {
  apply_external_forces(delta_t);
  mark_cells();
  rasterize();
  backup_velocity_field();
  apply_boundary_condition();
  compute_liquid_levelset();
  simple_extrapolate();
  project(delta_t);
  simple_extrapolate();
  advect(delta_t);
  t += delta_t;
}

void FLIPLiquid::reseed() {
}

void FLIPLiquid::correct_particle_positions(real delta_t, bool clear_c) {
  if (correction_strength == 0.0_f && !clear_c) {
    return;
  }
  NearestNeighbour2D nn;
  real range = 0.5f;
  std::vector<Vector2> positions;
  for (auto &p : particles) {
    positions.push_back(p.position);
  }
  nn.initialize(positions);
  std::vector<Vector2> delta_pos(particles.size());
  for (int i = 0; i < (int)particles.size(); i++) {
    delta_pos[i] = Vector2(0);
    auto &p = particles[i];
    std::vector<int> neighbour_index;
    std::vector<real> neighbour_dist;
    nn.query_n(p.position, correction_neighbours, neighbour_index,
               neighbour_dist);
    for (auto nei_index : neighbour_index) {
      if (nei_index == -1) {
        break;
      }
      auto &nei = particles[nei_index];
      real dist = length(p.position - nei.position);
      Vector2 dir = (p.position - nei.position) / dist;
      if (dist > 1e-4f && dist < range) {
        real a = correction_strength * delta_t * pow(1 - dist / range, 2);
        delta_pos[i] += a * dir;
        delta_pos[nei_index] -= a * dir;
      }
    }
    if (clear_c && (neighbour_index.size() <= 1 || neighbour_dist[1] > 1.5f)) {
      p.c[0] = p.c[1] = Vector2(0.0_f);
    }
  }
  for (int i = 0; i < (int)particles.size(); i++) {
    particles[i].position += delta_pos[i];
    clamp_particle(particles[i]);
  }
}

template <real (*T)(const Fluid::Particle &, const Vector2 &)>
void FLIPLiquid::rasterize_component(Array<real> &val, Array<real> &count) {
  val = 0;
  count = 0;
  real inv_kernel_size = 1.0_f / kernel_size;
  int extent = (kernel_size + 1) / 2;
  for (auto &p : particles) {
    for (auto &ind : val.get_rasterization_region(p.position, extent)) {
      Vector2 delta_pos = ind.get_pos() - p.position;
      real weight = kernel(inv_kernel_size * delta_pos);
      val[ind] += weight * T(p, delta_pos);
      count[ind] += weight;
    }
  }
  for (auto ind : val.get_region()) {
    if (count[ind] > 0) {
      val[ind] /= count[ind];
    }
  }
}

template void FLIPLiquid::rasterize_component<Fluid::Particle::get_velocity<0>>(
    Array<real> &val,
    Array<real> &count);
template void FLIPLiquid::rasterize_component<Fluid::Particle::get_velocity<1>>(
    Array<real> &val,
    Array<real> &count);
template void FLIPLiquid::rasterize_component<
    Fluid::Particle::get_affine_velocity<0>>(Array<real> &val,
                                             Array<real> &count);
template void FLIPLiquid::rasterize_component<
    Fluid::Particle::get_affine_velocity<1>>(Array<real> &val,
                                             Array<real> &count);

TC_IMPLEMENTATION(Fluid, FLIPLiquid, "flip_liquid");

TC_NAMESPACE_END
