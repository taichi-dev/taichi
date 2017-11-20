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
      ret = Vector(d[1], -d[0]);
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
      std::vector<Vector> vertices;
      for (int i = 0; i < 1000; i++) {
        std::string key_name = fmt::format("p{:03}", i);
        if (!config.has_key(key_name)) {
          break;
        }
        auto v = config.get<Vector>(key_name);
        vertices.push_back(v);
      }
      for (int i = 0; i < (int)vertices.size() - 1; i++) {
        Elem elem;
        elem.v[0] = vertices[i];
        elem.v[1] = vertices[i + 1];
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
