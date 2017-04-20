/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <taichi/math/array_3d.h>
#include <taichi/math/levelset_3d.h>
#include <taichi/math/dynamic_levelset_3d.h>

TC_NAMESPACE_BEGIN

class MPM3DScheduler {
protected:
    typedef Vector3 Vector;
    typedef Matrix3 Matrix;
    typedef Region3D Region;
public:
    static const int D = 3;

    // use mpm3d_grid_block_size for downsampling.
};

TC_NAMESPACE_END

