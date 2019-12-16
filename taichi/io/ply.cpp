/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/io/ply_writer.h>
#include <taichi/io/io.h>

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

// n, data_ptr, fn, name0, name1, name2, name3, ...
auto write_tcb_c_2 = [](const std::vector<std::string> &parameters) {
  auto n = (int)std::atoi(parameters[0].c_str());
  float *pos_ = reinterpret_cast<float *>(std::atol(parameters[1].c_str()));
  auto fn = parameters[2];
  using namespace taichi;
  WushiParticles data;
  for (int i = 0; i < (int)parameters.size() - 3; i++) {
    auto field_name = parameters[i + 3];
    auto &field = data[field_name];
    for (int j = 0; j < n; j++) {
      field.push_back(pos_[i * n + j]);
    }
  }
  write_to_binary_file(data, fn);
};

TC_REGISTER_TASK(write_tcb_c_2);

TC_NAMESPACE_END
