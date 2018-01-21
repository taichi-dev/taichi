/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <thread>
#include <set>
#include <mutex>
#include <vector>
#include <fstream>
#include <taichi/visualization/pakua.h>
#include <taichi/visualization/image_buffer.h>
#include <json.hpp>
using json = nlohmann::json;

TC_NAMESPACE_BEGIN

class JsonPakua : public Pakua {
  int frame_count;
  std::string frame_directory;

  json geometry;

 public:
  ~JsonPakua() {
  }

  void initialize(const Config &config) override {
    Pakua::initialize(config);
    frame_directory = config.get_string("frame_directory");
    frame_count = 0;
  }

  void add_point(Vector pos, Vector color, real size = 1.0f) override {
    for (int i = 0; i < 3; i++) {
      geometry["points"]["position"].push_back(pos[i]);
    }
    for (int i = 0; i < 3; i++) {
      geometry["points"]["color"].push_back(color[i]);
    }
    geometry["points"]["sizes"].push_back(size);
  }

  void add_line(const std::vector<Vector> &pos_v,
                const std::vector<Vector> &color_v,
                real width = 1.0f) override {
    /*
    int number = (int)pos_v.size();
    for (int i = 0; i < number; i++) {
      for (int j = 0; j < 3; j++)
        line_buffer.push_back(pos_v[i][j]);
      for (int j = 0; j < 3; j++)
        line_buffer.push_back(color_v[i][j]);
      line_buffer.push_back(width);
    }
    */
  }

  void add_triangle(const std::vector<Vector> &pos_v,
                    const std::vector<Vector> &color_v) override {
    assert(pos_v.size() == 3);
    assert(color_v.size() == 3);
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        geometry["triangles"]["position"].push_back(pos_v[i][k]);
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < 3; k++) {
        geometry["triangles"]["color"].push_back(color_v[i][k]);
      }
    }
  }

  void start() override {
    geometry["points"]["position"].clear();
    geometry["points"]["color"].clear();
    geometry["points"]["sizes"].clear();

    geometry["triangles"]["position"].clear();
    geometry["triangles"]["color"].clear();
  }

  void finish() override {
    std::ofstream f(fmt::format("{}/{:04}.json", frame_directory, frame_count));
    f << geometry;
    frame_count += 1;
  }

  Array2D<Vector3> screenshot() {
    return Array2D<Vector3>();
  }
};

TC_IMPLEMENTATION(Pakua, JsonPakua, "json");

TC_NAMESPACE_END
