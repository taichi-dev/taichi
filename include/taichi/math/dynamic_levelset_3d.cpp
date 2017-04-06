/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "dynamic_levelset_3d.h"

TC_NAMESPACE_BEGIN

void DynamicLevelSet3D::initialize(real _t0, real _t1, const LevelSet3D &_ls0, const LevelSet3D &_ls1) {
    t0 = _t0;
    t1 = _t1;
    levelset0 = std::make_shared<LevelSet3D>(_ls0);
    levelset1 = std::make_shared<LevelSet3D>(_ls1);
}

Vector3 DynamicLevelSet3D::get_spatial_gradient(const Vector3 &pos, real t) const {
    Vector3 gxyz0 = levelset0->get_gradient(pos);
    Vector3 gxyz1 = levelset1->get_gradient(pos);
    real gx = lerp((t - t0) / (t1 - t0), gxyz0.x, gxyz1.x);
    real gy = lerp((t - t0) / (t1 - t0), gxyz0.y, gxyz1.y);
    real gz = lerp((t - t0) / (t1 - t0), gxyz0.z, gxyz1.z);
    Vector3 gradient = Vector3(gx, gy, gz);
    if (length(gradient) < 1e-10f)
        return Vector3(1, 0, 0);
    else
        return normalize(gradient);
}

real DynamicLevelSet3D::get_temporal_derivative(const Vector3 &pos, real t) const {
    real l0 = levelset0->get(pos);
    real l1 = levelset1->get(pos);
    return (l1 - l0) / (t1 - t0);
}

    real DynamicLevelSet3D::sample(const Vector3 &pos, real t) const {
    real l1 = levelset0->get(pos);
    real l2 = levelset1->get(pos);
    return lerp((t - t0) / (t1 - t0), l1, l2);
}

Array3D<real> DynamicLevelSet3D::rasterize(int width, int height, int depth, real t) {
    Array3D<real> r0 = levelset0->rasterize(width, height, depth);
    Array3D<real> r1 = levelset1->rasterize(width, height, depth);
    Array3D<real> out(width, height, width);
    for (auto &ind : Region3D(0, width, 0, height, 0, depth, Vector3(0.5f, 0.5f, 0.5f))) {
        out[ind] = lerp((t - t0) / (t1 - t0), r0[ind], r1[ind]);
        if (std::isnan(out[ind])) {
            out[ind] = std::numeric_limits<real>::infinity();
        }
    }
    return out;
}


TC_NAMESPACE_END
