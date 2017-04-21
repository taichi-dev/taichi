/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <limits>
#include <taichi/math/array_2d.h>


TC_NAMESPACE_BEGIN

class LevelSet2D : public Array2D<real> {
public:
    static const real INF;

    real friction = 1.0f;

    LevelSet2D() : LevelSet2D(0, 0) {}

    LevelSet2D(int width, int height, Vector2 offset = Vector2(0.5f, 0.5f)) {
        initialize(width, height, offset);
    }

    void initialize(int width, int height, Vector2 offset) {
        Array2D<real>::initialize(width, height, INF, offset);
    }

    void initialize(const Vector2i &res, Vector2 offset) {
        Array2D<real>::initialize(res[0], res[1], INF, offset);
    }

    void initialize(int width, int height, Vector2 offset, real value) {
        Array2D<real>::initialize(width, height, value, offset);
    }

    void initialize(const Vector2i &res, Vector2 offset, real value) {
        Array2D<real>::initialize(res[0], res[1], value, offset);
    }

    std::shared_ptr<LevelSet2D> get_shared_ptr() {
        return std::make_shared<LevelSet2D>(*this);
    }

    void add_sphere(Vector2 center, real radius, bool inside_out = false);

    void add_polygon(std::vector<Vector2> polygon, bool inside_out = false);

    Vector2 get_gradient(const Vector2 &pos) const; // Note this is not normalized!

    Vector2 get_normalized_gradient(const Vector2 &pos) const;

    real get(const Vector2 &pos) const;

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

    Array2D<real> rasterize(int width, int height);
};

TC_NAMESPACE_END

