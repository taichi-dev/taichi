/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/interface.h>
#include <taichi/visualization/image_buffer.h>
#include <taichi/visual/camera.h>
#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

struct RenderParticle {
  Vector3 position;
  Vector4 color;
  RenderParticle() {
  }
  RenderParticle(const Vector3 &position, const Vector4 &color)
      : position(position), color(color) {
  }
  RenderParticle(const Vector3 &position, const Vector3 &color)
      : position(position), color(color, 1.0_f) {
  }
  bool operator==(const RenderParticle &p) const {
    // For boost::python vector_indexing_suite
    return false;
  }
};

class ParticleRenderer {
 protected:
  std::shared_ptr<Camera> camera;

 public:
  void set_camera(std::shared_ptr<Camera> camera) {
    this->camera = camera;
  }
  virtual void initialize(const Config &config){};
  virtual void render(Array2D<Vector3> &buffer,
                      const std::vector<RenderParticle> &particles) const {
  }
};

std::shared_ptr<Texture> rasterize_render_particles(
    const Config &config,
    const std::vector<RenderParticle> &particles);

TC_INTERFACE(ParticleRenderer)

TC_NAMESPACE_END
