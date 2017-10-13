/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/visualization/image_buffer.h>
#include <taichi/common/interface.h>
#include <taichi/math/array_2d.h>
#include <taichi/math/levelset.h>

TC_NAMESPACE_BEGIN

class Fluid {
 public:
  template <typename T>
  using Array = Array2D<T>;

  struct Particle {
    Vector3 color = Vector3(-1, 0, 0);
    Vector2 position, velocity;
    Vector2 weight;
    Vector2 c[2] = {Vector2(0), Vector2(0)};  // for APIC
    long long id = instance_counter++;
    real temperature;
    real radius = 0.75f;

    Particle(){};

    Particle(Vector2 position, Vector2 velocity = Vector2(0))
        : position(position), velocity(velocity) {
    }

    void move(Vector2 delta_x) {
      this->position += delta_x;
    }

    template <int k>
    static real get_velocity(const Particle &p, const Vector2 &delta_pos) {
      return p.velocity[k];
    }

    template <int k>
    static real get_affine_velocity(const Particle &p,
                                    const Vector2 &delta_pos) {
      return p.velocity[k] + dot(p.c[k], delta_pos);
    }

    static real get_signed_distance(const Particle &p,
                                    const Vector2 &delta_pos) {
      return length(delta_pos) - p.radius;
    }

    static long long instance_counter;

    bool operator==(const Particle &o) {
      return o.id == id;
    }
  };

  virtual void initialize(const Config &config) {
  }

  virtual void set_levelset(const LevelSet2D &boundary_levelset) {
  }

  virtual void step(real delta_t) {
  }

  virtual real get_current_time() {
    return 0.0_f;
  }

  virtual void add_particle(Particle &particle) {
  }

  virtual std::vector<Particle> get_particles() {
    return std::vector<Particle>();
  }

  virtual LevelSet2D get_liquid_levelset() {
    return LevelSet2D();
  }

  virtual Array<real> get_density() {
    return Array<real>(Vector2i(0));
  }

  virtual void add_source(const Config &config) {
  }

  virtual Array<real> get_pressure() {
    return Array<real>(Vector2i(0, 0));
  }

 protected:
  std::vector<Particle> particles;
};

TC_INTERFACE(Fluid);

TC_NAMESPACE_END
