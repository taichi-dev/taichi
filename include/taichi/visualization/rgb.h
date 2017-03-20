/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <glm/glm.hpp>

#include <taichi/common/util.h>
#include <taichi/math/math_util.h>

TC_NAMESPACE_BEGIN
#undef RGB

class RGB {
public:
    real r, g, b;

    RGB() {
        r = g = b = 0.0;
    }

    RGB(real r, real g, real b) : r(r), g(g), b(b) { }

    operator glm::vec3() {
        return glm::vec3(r / 255.0f, g / 255.0f, b / 255.0f);
    }

    void append_to_string(std::string &str) {
        str.push_back((char)int(clamp(r, 0.0f, 1.0f) * 255.0));
        str.push_back((char)int(clamp(g, 0.0f, 1.0f) * 255.0));
        str.push_back((char)int(clamp(b, 0.0f, 1.0f) * 255.0));
    }
};

TC_NAMESPACE_END
