/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/util.h>
#include <taichi/math.h>

TC_NAMESPACE_BEGIN

class Pakua : Unit {
 public:
  using Vector2 = VectorND<2, real>;
  using Vector3 = VectorND<3, real>;
  using Vector = VectorND<3, real>;

  virtual void initialize(const Config &config) override {
    //int port = config.get<int32>("port");
  }

  // Add a particle to buffer
  virtual void add_point(Vector pos, Vector color, real size = 1.0_f) = 0;

  virtual void add_point(Vector2 pos, Vector color, real size = 1.0_f) {
    this->add_point(Vector3(pos, 0.5f), color, size);
  }

  // Add a line to buffer
  virtual void add_line(const std::vector<Vector3> &pos,
                        const std::vector<Vector3> &color,
                        real width = 1.0_f) = 0;

  virtual void add_line(const std::vector<Vector2> &pos_,
                        const std::vector<Vector3> &color,
                        real width = 1.0_f) {
    std::vector<Vector3> pos;
    for (auto &p : pos_) {
      pos.push_back(Vector3(p, 0.5f));
    }
    this->add_line(pos, color);
  }

  // Add a triangle to buffer
  virtual void add_triangle(const std::vector<Vector3> &pos,
                            const std::vector<Vector3> &color) = 0;

  virtual void add_triangle(const std::vector<Vector2> &pos_,
                            const std::vector<Vector3> &color) {
    std::vector<Vector3> pos;
    for (auto &p : pos_) {
      pos.push_back(Vector3(p, 0.5f));
    }
    this->add_triangle(pos, color);
  }

  // Reset and start a new canvas
  virtual void start() = 0;

  // Finish and send canvas (buffer) to frontend
  virtual void finish() = 0;

  virtual void set_resolution(Vector2i res) {NOT_IMPLEMENTED};
};

TC_INTERFACE(Pakua)

TC_NAMESPACE_END
