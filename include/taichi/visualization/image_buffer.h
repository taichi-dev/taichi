/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <taichi/math/math.h>
#include <taichi/math/array_2d.h>
#include <taichi/system/threading.h>
#include <stb_image.h>
#include <stb_image_write.h>

TC_NAMESPACE_BEGIN

template <typename T>
class ImageAccumulator {
 public:
  std::vector<Spinlock> locks;

  ImageAccumulator() {}

  ImageAccumulator(Vector2i res) : buffer(res), counter(res), res(res) {
    for (int i = 0; i < res[0] * res[1]; i++) {
      locks.push_back(Spinlock());
    }
  }

  Array2D<T> get_averaged(T default_value = T(0)) {
    Array2D<T> result(res);
    for (int i = 0; i < res[0]; i++) {
      for (int j = 0; j < res[1]; j++) {
        if (counter[i][j] > 0) {
          real inv = (real)1 / counter[i][j];
          result[i][j] = inv * buffer[i][j];
        } else {
          result[i][j] = default_value;
        }
      }
    }
    return result;
  }

  void accumulate(int x, int y, T val) {
    int lock_id = x * res[1] + y;
    locks[lock_id].lock();
    counter[x][y]++;
    buffer[x][y] += val;
    locks[lock_id].unlock();
  }

  void accumulate(ImageAccumulator<T> &other) {
    for (int i = 0; i < res[0]; i++) {
      for (int j = 0; j < res[1]; j++) {
        counter[i][j] += other.counter[i][j];
        buffer[i][j] += other.buffer[i][j];
      }
    }
  }

  int get_width() const { return res[0]; }

  int get_height() const { return res[1]; }

 private:
  Array2D<T> buffer;
  Array2D<int> counter;
  Vector2i res;
};

TC_NAMESPACE_END
