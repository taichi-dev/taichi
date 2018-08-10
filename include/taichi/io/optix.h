/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#pragma once
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

struct OptiXMesh {
  struct Vertex {
    Vector3 position;
    Vector3 normal;
    Vector3 tangent;
    Vector2 uv;
    TC_IO_DEF(position, normal, tangent, uv);
  };
  std::vector<Vertex> vertices;
  std::vector<std::array<uint32, 3>> faces;
  TC_IO_DEF(vertices, faces);

  void recompute_normals() {
    for (auto &v : vertices) {
      v.normal = Vector3(0);
    }
    for (auto &f : faces) {
      int a = f[0], b = f[1], c = f[2];
      vertices[a].normal += cross(vertices[b].position - vertices[a].position,
                                  vertices[c].position - vertices[a].position);
    }
    for (auto &v : vertices) {
      if (length2(v.normal) != 0) {
        v.normal = normalized(v.normal);
      } else {
        v.normal = Vector3(0, 1, 0);
      }
      if (std::abs(v.normal.y > 0.9_f)) {
        v.tangent = normalized(cross(v.normal, Vector3(0, 1, 0)));
      } else {
        v.tangent = normalized(cross(v.normal, Vector3(1, 0, 0)));
      }
    }
  }
};

struct OptiXParticle {
  Vector4 position_and_radius;
  TC_IO_DEF(position_and_radius);
};

struct OptiXScene {
  std::vector<OptiXMesh> meshes;
  std::vector<OptiXParticle> particles;
  TC_IO_DEF(meshes, particles);
};

TC_NAMESPACE_END