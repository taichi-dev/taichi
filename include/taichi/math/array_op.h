/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/array_2d.h>
#include <taichi/math/math_util.h>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

template <typename T>
Array2D<T> gaussian_blur_x(const Array2D<T> &arr, real sigma) {
    if (sigma < 1e-5f) {
        return arr;
    }
    Array2D<T> ret = arr.same_shape(T(0.0f));
    int radius = int(std::ceil(sigma)) * 3;
    std::vector<real> stencil(radius + 1);

    real tot = 0.0f;
    for (int i = 0; i <= radius; i++) {
        stencil[i] = std::exp(-0.5f * i * i / sigma / sigma);
        tot += stencil[i] * (1 + (i != 0));
    }
    for (int i = 0; i <= radius; i++) {
        stencil[i] /= tot;
    }

    // X dir
    for (auto &ind : arr.get_region()) {
        T tot = stencil[0] * arr[ind];
        for (int k = 1; k <= radius; k++) {
            tot += stencil[k] * (arr[std::max(0, ind.i - k)][ind.j]
                + arr[std::min(ind.i + k, arr.get_width() - 1)][ind.j]);
        }
        ret[ind] = tot;
    }
    return ret;
}

template <typename T>
Array2D<T> gaussian_blur_y(const Array2D<T> &arr, real sigma) {
    if (sigma < 1e-5f) {
        return arr;
    }
    Array2D<T> ret = arr.same_shape(T(0.0f));
    int radius = int(std::ceil(sigma)) * 3;
    std::vector<real> stencil(radius + 1);

    real tot = 0.0f;
    for (int i = 0; i <= radius; i++) {
        stencil[i] = std::exp(-0.5f * i * i / sigma / sigma);
        tot += stencil[i] * (1 + (i != 0));
    }
    for (int i = 0; i <= radius; i++) {
        stencil[i] /= tot;
    }

    // Y dir
    for (auto &ind : arr.get_region()) {
        T tot = stencil[0] * arr[ind];
        for (int k = 1; k <= radius; k++) {
            tot += stencil[k] * (arr[ind.i][std::max(0, ind.j - k)] + 
                arr[ind.i][std::min(ind.j + k, arr.get_height() - 1)]);
        }
        ret[ind] = tot;
    }
    return ret;
}

template <typename T>
Array2D<T> gaussian_blur(const Array2D<T> &arr, real sigma) {
    return gaussian_blur_x(gaussian_blur_y(arr, sigma), sigma);
}

template<typename T>
Array2D<T> take_downsampled(const Array2D<T> &arr, int step) {
    Array2D<T> ret(arr.get_width() / step, arr.get_height() / step);
    for (auto &ind : ret.get_region()) {
        ret[ind] = arr[ind.i * step][ind.j * step];
    }
    return ret;
}

TC_NAMESPACE_END

