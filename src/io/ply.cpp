/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/io/ply_writer.h>

TC_NAMESPACE_BEGIN

class TestPLY : public Task {
  std::string run() override {
    PLYWriter ply("/tmp/test.ply");

    using Vert = PLYWriter::Vertex;

    for (int i = 0; i < 10; i++) {
      ply.add_face({Vert(Vector3(1 + i, 0, 0), Vector3(1, 0, 0)),
                    Vert(Vector3(0, 1 + i, 0), Vector3(0, 1, 0)),
                    Vert(Vector3(0, 0, 1 + i), Vector3(0, 0, i % 5 * 0.2f))});
    }
    return "";
  }
};

TC_IMPLEMENTATION(Task, TestPLY, "test_ply");

TC_NAMESPACE_END
