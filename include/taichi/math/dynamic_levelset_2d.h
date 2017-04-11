/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once


#include <limits>
#include <memory>
#include <taichi/math/levelset_2d.h>


TC_NAMESPACE_BEGIN

class DynamicLevelSet2D {
public:
    real t0, t1;
    std::shared_ptr<LevelSet2D> levelset0, levelset1;

    void initialize(real _t0, real _t1, const LevelSet2D &_ls0, const LevelSet2D &_ls1);

    Vector2 get_spatial_gradient(const Vector2 &pos, real t) const;

    real get_temporal_derivative(const Vector2 &pos, real t) const;

    real sample(const Vector2 &pos, real t) const;

    Array2D<real> rasterize(int width, int height, real t);
};

TC_NAMESPACE_END

