/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

class SphericalGradientTexture final : public Texture {
 private:
  Vector4 inside_val;
  Vector4 outside_val;
  real angle, sharpness;

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    TC_LOAD_CONFIG(inside_val, Vector4(0.0_f));
    TC_LOAD_CONFIG(outside_val, Vector4(1.0_f));
    TC_LOAD_CONFIG(angle, 30.0_f);
    TC_LOAD_CONFIG(sharpness, 1.0_f);
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    real theta_d = (coord.x * 2 + 1) * pi, phi_d = (coord.y - 0.5f) * pi;
    real angle_ =
        std::acos(std::sin(theta_d) * std::cos(phi_d)) * (180.0_f / pi);
    real t = std::tanh((angle_ - angle) * sharpness * 0.01f) * 0.5f + 0.5f;
    return lerp(t, inside_val, outside_val);
  }
};

TC_IMPLEMENTATION(Texture, SphericalGradientTexture, "spherical_gradient");

TC_NAMESPACE_END
