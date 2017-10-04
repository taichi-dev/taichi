/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <taichi/math/math.h>
#include <taichi/visual/sampler.h>
#include <taichi/geometry/primitives.h>

TC_NAMESPACE_BEGIN

enum class VolumeEvent { scattering, absorption };

class VolumeMaterial : public Unit {
 protected:
  Matrix4 world2local;
  Matrix4 local2world;

 public:
  VolumeMaterial() { set_transform(Matrix4(1.0_f)); }
  virtual void initialize(const Config &config) {
    printf(
        "Info: Volumetric rendering is turned ON. Note that PT & MCMCPT are "
        "only renderers that support this.\n");
    printf("      This may lead to different output by different renderers.\n");
    this->volumetric_scattering = config.get<real>("scattering");
    this->volumetric_absorption = config.get<real>("absorption");
    if (config.has_key("transform_ptr")) {
      set_transform(*config.get_ptr<Matrix4>("transform_ptr"));
    }
  }

  void set_transform(const Matrix4 &local2world) {
    this->local2world = local2world;
    this->world2local = inverse(local2world);
  }

  virtual real sample_free_distance(StateSequence &rand, const Ray &ray) const {
    real kill = volumetric_scattering + volumetric_absorption;
    if (kill > 0) {
      return -log(1 - rand()) / kill;
    } else {
      return std::numeric_limits<real>::infinity();
    }
  }

  virtual real unbiased_sample_attenuation(const Vector3 &start,
                                           const Vector3 &end,
                                           StateSequence &rand) const {
    return get_attenuation(
        length(multiply_matrix4(world2local, end - start, 0)));
  }

  virtual VolumeEvent sample_event(StateSequence &rand, const Ray &ray) const {
    return rand() < volumetric_scattering /
                        (volumetric_scattering + volumetric_absorption)
               ? VolumeEvent::scattering
               : VolumeEvent::absorption;
  }

  virtual Vector3 sample_phase(StateSequence &rand, const Ray &ray) const {
    return sample_sphere(rand(), rand());
  }

  virtual Vector3 phase_evaluate(const Vector3 &pos,
                                 const Vector3 &in_dir,
                                 const Vector3 &out_dir) {
    return Vector3(1.0_f, 1.0_f, 1.0_f);
  }

  virtual real phase_probability_density(const Vector3 &pos,
                                         const Vector3 &in_dir,
                                         const Vector3 &out_dir) {
    return 1 / 4 / pi;
  }

  virtual bool is_vacuum() const { return false; }

 protected:
  virtual real get_attenuation(real dist) const {
    return exp(-dist * (volumetric_scattering + volumetric_absorption));
  }

  real volumetric_scattering;
  real volumetric_absorption;
};

TC_INTERFACE(VolumeMaterial);

class VolumeStack {
  std::vector<VolumeMaterial const *> stack;

 public:
  VolumeStack();
  void push(VolumeMaterial const *vol) { stack.push_back(vol); }
  void pop() { stack.pop_back(); }
  VolumeMaterial const *top() { return stack.back(); }

  size_t size() const { return stack.size(); }
};

class VolumeStackPushGuard {
 private:
  VolumeStack &stack;

 public:
  VolumeStackPushGuard(VolumeStack &stack, const VolumeMaterial &volume)
      : stack(stack) {
    stack.push(&volume);
  }
  ~VolumeStackPushGuard() { stack.pop(); }
};
TC_NAMESPACE_END
