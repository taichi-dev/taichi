/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

inline float catmull_rom(float f_m_1, float f_0, float f_1, float f_2,
                         float x_r) {
    // return (1 - x_r) * f_0 + x_r * f_1;
    float s = (f_1 - f_0);
    float s_0 = (f_1 - f_m_1) / 2.0f;
    float s_1 = (f_2 - f_0) / 2.0f;
    s_0 = s_0 * (s_1 * s > 0);
    s_1 = s_1 * (s_1 * s > 0);
    return f_0 + x_r * s_0 + (-3 * f_0 + 3 * f_1 - 2 * s_0 - s_1) * x_r * x_r +
           (2 * f_0 - 2 * f_1 + s_0 + s_1) * x_r * x_r * x_r;
}

inline float catmull_rom(float *pf_m_1, float x_r) {
    return catmull_rom(*pf_m_1, *(pf_m_1 + 1), *(pf_m_1 + 2), *(pf_m_1 + 3), x_r);
}

TC_NAMESPACE_END

