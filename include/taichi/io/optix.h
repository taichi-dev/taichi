/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

struct OptiXMesh {
  struct Vertex {
    Vector3 position;
    Vector3 normal;
    Vector2 uv;
    TC_IO_DEF(position, normal, uv);
  };
  std::vector<Vertex> vertices;
  std::vector<std::array<uint32, 3>> faces;
  TC_IO_DEF(vertices, faces);
};

TC_NAMESPACE_END