/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/
#include <taichi/util.h>

TC_NAMESPACE_BEGIN

class PLYWriter {
 public:
  FILE *file;

  struct Vertex {
    Vector3 position;
    Vector3 color;
    Vertex(Vector3 position, Vector3 color) : position(position), color(color) {
    }
  };

  std::vector<Vertex> vertices;
  std::vector<int> face_vertices_count;

  PLYWriter(const std::string &file_name) {
    file = fopen(file_name.c_str(), "w");
  }

  void add_face(const std::vector<Vertex> &vert) {
    vertices.insert(vertices.end(), vert.begin(), vert.end());
    face_vertices_count.push_back(vert.size());
  }

  void add_face(const std::vector<Vector3> &vert_) {
    std::vector<Vertex> vert;
    for (auto v : vert_) {
      vert.push_back(Vertex(v, Vector3(0, 1, 0)));
    }
    vertices.insert(vertices.end(), vert.begin(), vert.end());
    face_vertices_count.push_back(vert.size());
  }

  ~PLYWriter() {
    std::string header =
        "ply\n"
        "format ascii 1.0\n"
        "element vertex {}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "element face {}\n"
        "property list uchar int vertex_index\n"
        "end_header\n";

    fmt::print(file, header, vertices.size(), face_vertices_count.size());

    for (auto v : vertices) {
      Vector3i color = (v.color * 255.0_f).template cast<int>();
      fmt::print(file, "{} {} {} {} {} {}\n", v.position.x, v.position.y,
                 v.position.z, color.x, color.y, color.z);
    }

    int begin, end = 0;
    for (auto count : face_vertices_count) {
      begin = end;
      end += count;
      fmt::print(file, "{} ", count);
      for (int i = begin; i < end; i++) {
        fmt::print(file, "{} ", i);
      }
      fmt::print(file, "\n");
    }
    fclose(file);
  }
};

TC_NAMESPACE_END