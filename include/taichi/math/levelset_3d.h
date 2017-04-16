/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <limits>
#include <taichi/math/array_3d.h>


TC_NAMESPACE_BEGIN

class LevelSet3D : public Array3D<real> {
public:
    static const real INF;

    real friction = 1.0f;

    LevelSet3D() : LevelSet3D(0, 0, 0) {}

    LevelSet3D(int width, int height, int depth, Vector3 offset = Vector3(0.5f, 0.5f, 0.5f)) {
        initialize(width, height, depth, offset);
    }

    void initialize(int width, int height, int depth, Vector3 offset) {
        Array3D<real>::initialize(width, height, depth, INF, offset);
    }

    void initialize(const Vector3i &res, Vector3 offset) {
        Array3D<real>::initialize(res[0], res[1], res[2], INF, offset);
    }

    void initialize(int width, int height, int depth, Vector3 offset, real value) {
        Array3D<real>::initialize(width, height, depth, value, offset);
    }

    void initialize(const Vector3i &res, Vector3 offset, real value) {
        Array3D<real>::initialize(res[0], res[1], res[2], value, offset);
    }

    void add_sphere(Vector3 center, real radius, bool inside_out = false);

    Vector3 get_gradient(const Vector3 &pos) const; // Note this is not normalized!

    Vector3 get_normalized_gradient(const Vector3 &pos) const;

    real get(const Vector3 &pos) const;

    static real fraction_outside(real phi_a, real phi_b) {
        return 1.0f - fraction_inside(phi_a, phi_b);
    }

    static real fraction_inside(real phi_a, real phi_b) {
        if (phi_a < 0 && phi_b < 0)
            return 1;
        if (phi_a < 0 && phi_b >= 0)
            return phi_a / (phi_a - phi_b);
        if (phi_a >= 0 && phi_b < 0)
            return phi_b / (phi_b - phi_a);
        else
            return 0;
    }

    Array3D<real> rasterize(int width, int height, int depth);
};

TC_NAMESPACE_END

