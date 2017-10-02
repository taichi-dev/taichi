/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/array_2d.h>
#include <taichi/math/math.h>
#include <taichi/math/math.h>

TC_NAMESPACE_BEGIN

template <int N, typename T>
ArrayND<N, T> symmetric_convolution(const ArrayND<N, T> &arr,
                                    std::vector<real> kernel,
                                    int axis,
                                    bool normalize = true) {
  ArrayND<N, T> ret = arr.same_shape(T(0.0_f));
  const int radius = kernel.size() - 1;
  if (normalize) {
    real tot(0.0);
    for (int i = 0; i <= radius; i++) {
      tot += kernel[i] * (1 + (i != 0));
    }
    for (int i = 0; i <= radius; i++) {
      kernel[i] /= tot;
    }
  }
  for (auto &ind : arr.get_region()) {
    T tot(0);
    for (int k = -radius; k <= radius; k++) {
      VectorND<N, int> acc = ind.get_ipos();
      acc[axis] = clamp(acc[axis] + k, 0, arr.get_res()[axis] - 1);
      tot += kernel[std::abs(k)] * arr[acc];
    }
    ret[ind] = tot;
  }
  return ret;
}

template <typename T>
Array2D<T> box_blur(const Array2D<T> &arr, real radius, int axis = -1) {
  auto kernel = std::vector<real>(radius, 1.0_f);
  if (axis == -1) {
    return box_blur(box_blur(arr, radius, 0), radius, 1);
  } else {
    return symmetric_convolution(arr, kernel, axis);
  }
}

template <int N, typename T>
ArrayND<N, T> gaussian_blur(const ArrayND<N, T> &arr,
                            real sigma,
                            int axis = -1) {
  if (sigma < 1e-5f) {
    return arr;
  }

  if (axis == -1) {
    auto output = arr;
    for (int i = 0; i < N; i++) {
      output = gaussian_blur(output, sigma, i);
    }
    return output;
  } else {
    int radius = int(std::ceil(sigma * 3.0f));
    std::vector<real> stencil(radius + 1);
    for (int i = 0; i <= radius; i++)
      stencil[i] = std::exp(-0.5f * i * i / sigma / sigma);
    return symmetric_convolution(arr, stencil, axis);
  }
}

template <typename T>
Array2D<T> take_downsampled(const Array2D<T> &arr, int step) {
  Array2D<T> ret(Vector2i(arr.get_width() / step, arr.get_height() / step));
  for (auto &ind : ret.get_region()) {
    ret[ind] = arr[ind.i * step][ind.j * step];
  }
  return ret;
}

TC_NAMESPACE_END
