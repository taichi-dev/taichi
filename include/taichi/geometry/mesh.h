/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <taichi/common/config.h>
#include <taichi/math.h>
#include <taichi/visual/scene.h>

TC_NAMESPACE_BEGIN

template <int dim>
struct Element {
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;
  using Matrix = MatrixND<dim, real>;
  using MatrixP = MatrixND<dim + 1, real>;

  Vector v[dim];
  bool open_end[dim];

  Element() {
    for (int i = 0; i < dim; i++) {
      v[i] = Vector(0.0_f);
      open_end[i] = false;
    }
  }

  Element get_transformed(const MatrixP &m) const {
    Element ret;
    for (int i = 0; i < dim; i++) {
      ret.v[i] = transform(m, v[i], 1);
      ret.open_end[i] = open_end[i];
    }
    return ret;
  }

  Vector get_center() const {
    Vector sum(0);
    for (int i = 0; i < dim; i++) {
      sum += v[i];
    }
    return 1.0_f / dim * sum;
  }

  Vector get_normal() const {
    Vector ret;
    TC_STATIC_IF(dim == 2) {
      Vector d = v[1] - v[0];
      ret = normalized(Vector(d[1], -d[0]));
    }
    TC_STATIC_ELSE {
      Vector n = cross(v[1] - v[0], v[2] - v[1]);
      ret = normalized(n);
    }
    TC_STATIC_END_IF
    return ret;
  }

  TC_IO_DECL {
    TC_IO(v);
    TC_IO(open_end);
  }
};

template <int dim>
struct ElementMesh {
  using Elem = Element<dim>;
  std::vector<Elem> elements;
  using Vector = VectorND<dim, real>;
  using Vectori = VectorND<dim, int>;
  using MatrixP = MatrixND<dim + 1, real>;

  void initialize(const Config &config) {
    TC_STATIC_IF(dim == 2) {
      std::string s = config.get<std::string>("segment_mesh");
      std::stringstream ss(s);
      int n;
      ss >> n;
      for (int i = 0; i < n; i++) {
        Elem elem;
        ss >> elem.v[0][0];
        ss >> elem.v[0][1];
        ss >> elem.v[1][0];
        ss >> elem.v[1][1];
        elements.push_back(elem);
      }
    }
    TC_STATIC_ELSE {
      TC_INFO("Adding mesh, fn={}", config.get<std::string>("mesh_fn"));
      std::string mesh_fn = config.get<std::string>("mesh_fn");
      std::string full_fn = std::getenv("TAICHI_ROOT_DIR") +
                            std::string("/taichi/projects/mpm/data/") + mesh_fn;
      std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
      Config mesh_config;
      mesh_config.set("filename", full_fn);
      mesh_config.set("reverse_vertices",
                      config.get<bool>("reverse_vertices", false));
      mesh->initialize(mesh_config);
      for (auto tri : mesh->get_triangles()) {
        Elem elem;
        for (int i = 0; i < 3; i++) {
          elem.v[i] = tri.v[i];
        }
        elements.push_back(elem);
      }
    }
    TC_STATIC_END_IF
  }
};

TC_NAMESPACE_END
