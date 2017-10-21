/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "dcraw.h"

#include <mutex>
#include <taichi/io/image_reader.h>

TC_NAMESPACE_BEGIN

Array2D<Vector4> dcraw_read(const std::string &filepath) {
  // Single threaded...
  static std::mutex lock;
  std::lock_guard<std::mutex> lock_guard(lock);
  std::string filepath_non_const = filepath;
  std::vector<const char *> argv{"dcraw.exe", "-4", "-T", "-W",
                                 filepath_non_const.c_str()};
  DCRawOutput output;
  dcraw_main((int)argv.size(), &argv[0], output);
  auto img =
      Array2D<Vector4>(Vector2i(output.width, output.height), Vector4(0.0_f));
  for (auto &ind : img.get_region()) {
    for (int i = 0; i < output.channels; i++) {
      img[ind][i] =
          output.data[output.channels * (ind.j * output.width + ind.i) + i];
    }
  }
  delete[] output.data;
  img.flip(1);
  TC_P(output.width);
  TC_P(output.height);
  TC_P(img.get_width());
  TC_P(img.get_height());
  TC_P(img.get_size());
  return img;
}

class RawImageReader final : public ImageReader {
 public:
  void initialize(const Config &config) override {
  }

  Array2D<Vector4> read(const std::string &filepath) override {
    auto tmp = dcraw_read(filepath);
    TC_P("read!!");
    return tmp;
  }
};

TC_IMPLEMENTATION(ImageReader, RawImageReader, "raw");

TC_NAMESPACE_END