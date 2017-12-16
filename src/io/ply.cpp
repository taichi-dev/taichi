/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/io/ply_writer.h>

TC_NAMESPACE_BEGIN

class TestPLY : public Task {
  void run() override {
    PLYWriter ply("/tmp/test.ply");

    using Vert = PLYWriter::Vertex;

    for (int i = 0; i < 10; i++) {
      ply.add_face({Vert(Vector3f(1 + i, 0, 0), Vector3f(1, 0, 0)),
                    Vert(Vector3f(0, 1 + i, 0), Vector3f(0, 1, 0)),
                    Vert(Vector3f(0, 0, 1 + i), Vector3f(0, 0, i % 5 * 0.2f))});
    }
  }
};

TC_IMPLEMENTATION(Task, TestPLY, "test_ply");

TC_NAMESPACE_END
