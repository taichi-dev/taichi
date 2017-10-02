/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "operations.h"

TC_NAMESPACE_BEGIN

int64 binomial(int n, int r) {
  int64 ret = 1;
  for (int i = 1; i <= r; i++) {
    ret = ret * (n - i + 1) / i;
  }
  return ret;
}

Array2D<Vector3> blur_with_depth(const Array2D<Vector3> &image,
                                 const Array2D<Vector3> &depth,
                                 int filter_type,
                                 real focal_plane,
                                 real aperature) {
  auto ret = image.same_shape();
  constexpr int max_radius = 50;

  std::vector<real> kernel(max_radius, 0.0f);

  for (auto &ind : image.get_region()) {
    int radius;
    real sigma = aperature * std::abs(depth[ind].x - focal_plane) + 1e-7f;
    if (filter_type == 0) {
      // Gaussian
      radius = int(std::ceil(sigma * 3.0f));
      for (int i = 0; i <= radius; i++)
        kernel[i] = std::exp(-0.5f * i * i / sigma / sigma);
    } else {
      // Binomial
      radius = (int)(std::round(sigma) / 2);
      for (int i = 0; i <= radius; i++)
        kernel[i] = binomial(radius * 2 + 1, radius - i);
    }
    // normalize kernel
    real kernel_tot(0.0);
    for (int i = 0; i <= radius; i++) {
      kernel_tot += kernel[i] * (1 + (i != 0));
    }
    for (int i = 0; i <= radius; i++) {
      kernel[i] /= kernel_tot;
    }
    Vector3 tot = 0;
    for (int i = -radius; i <= radius; i++) {
      for (int j = -radius; j <= radius; j++) {
        Vector2i coord(clamp(ind.i + i, 0, image.get_width() - 1),
                       clamp(ind.j + j, 0, image.get_height() - 1));
        tot += kernel[std::abs(i)] * kernel[std::abs(j)] * image[coord];
      }
    }
    ret[ind] = tot;
  }
  return ret;
}

Array2D<Vector3> seam_carving(const Array2D<Vector3> &image) {
  Array2D<Vector3> carved(Vector2i(image.get_width() - 1, image.get_height()));

  real minimum_energy = std::numeric_limits<real>::infinity();
  int minimum_energy_column = -1;

  for (int i = 1; i < image.get_width() - 1; i++) {
    real total_energy = 0.0f;
    for (int j = 1; j < image.get_height() - 1; j++) {
      for (int k = 0; k < 3; k++) {
        real grad_x = image[i + 1][j + 1][k] + image[i + 1][j][k] * 2 +
                      image[i + 1][j - 1][k] - image[i - 1][j + 1][k] -
                      image[i - 1][j][k] * 2 - image[i - 1][j - 1][k];
        real grad_y = image[i + 1][j + 1][k] + image[i][j + 1][k] * 2 +
                      image[i][j - 1][k] - image[i + 1][j + 1][k] -
                      image[i][j + 1][k] * 2 - image[i][j - 1][k];
        total_energy += pow<2>(grad_x) + pow<2>(grad_y);
      }
    }
    if (total_energy < minimum_energy) {
      minimum_energy = total_energy;
      minimum_energy_column = i;
    }
  }

  for (auto &ind : carved.get_region()) {
    carved[ind] = image[ind.get_ipos() +
                        Vector2i(int(ind.i >= minimum_energy_column), 0)];
  }

  return carved;
}

TC_NAMESPACE_END