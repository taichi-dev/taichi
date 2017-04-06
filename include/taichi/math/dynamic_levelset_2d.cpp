/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "dynamic_levelset_2d.h"

TC_NAMESPACE_BEGIN

void DynamicLevelSet2D::initialize(real _t0, real _t1, const LevelSet2D &_ls0, const LevelSet2D &_ls1) {
    t0 = _t0;
    t1 = _t1;
    levelset0 = std::make_shared<LevelSet2D>(_ls0);
    levelset1 = std::make_shared<LevelSet2D>(_ls1);
}

Vector2 DynamicLevelSet2D::get_spatial_gradient(const Vector2 &pos, real t) const {
    Vector2 gxy0 = levelset0->get_gradient(pos);
    Vector2 gxy1 = levelset1->get_gradient(pos);
    real gx = lerp((t - t0) / (t1 - t0), gxy0.x, gxy1.x);
    real gy = lerp((t - t0) / (t1 - t0), gxy0.y, gxy1.y);
    Vector2 gradient = Vector2(gx, gy);
    if (length(gradient) < 1e-10f)
        return Vector2(1, 0);
    else
        return normalize(gradient);
}

real DynamicLevelSet2D::get_temporal_derivative(const Vector2 &pos, real t) const {
    real l1 = levelset0->get(pos);
    real l0 = levelset1->get(pos);
    return (l1 - l0) / (t1 - t0);
}

real DynamicLevelSet2D::sample(const Vector2 &pos, real t) const {
    real l1 = levelset0->sample(pos);
    real l2 = levelset1->sample(pos);
    return lerp((t - t0) / (t1 - t0), l1, l2);
}

Array2D<real> DynamicLevelSet2D::rasterize(int width, int height, real t) {
    Array2D<real> r0 = levelset0->rasterize(width, height);
    Array2D<real> r1 = levelset1->rasterize(width, height);
    Array2D<real> out(width, height);
    for (auto &ind : Region2D(0, width, 0, height, Vector2(0.5f, 0.5f))) {
        out[ind] = lerp((t - t0) / (t1 - t0), r0[ind], r1[ind]);
        if (std::isnan(out[ind])) {
            out[ind] = std::numeric_limits<real>::infinity();
        }
    }
    return out;
}


TC_NAMESPACE_END
