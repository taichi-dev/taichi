/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math_util.h>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

void imp_svd(const Matrix3 &m, Matrix3 &u, Matrix3 &s, Matrix3 &v);

TC_NAMESPACE_END
