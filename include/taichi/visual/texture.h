/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/linalg.h>
#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

// I guess looking up a pointer can be more costy than returning 1 or 2 more floats, thus let's just
// treat Textued2D as special case of Texture3D.
// Maybe I'll do it better when I have time.

class Texture : public Unit {
public:
    virtual void initialize(const Config &config) {}
    virtual Vector4 sample(const Vector2 &coord) const {return sample(Vector3(coord.x, coord.y, 0.5f));}
    virtual Vector4 sample(const Vector3 &coord) const {error("no impl"); return Vector4(0.0f);}
    Vector3 sample3(const Vector2 &coord) const {
        Vector4 tmp = sample(coord);
        return Vector3(tmp.x, tmp.y, tmp.z);
    }
    Vector3 sample3(const Vector3 &coord) const {
        Vector4 tmp = sample(coord);
        return Vector3(tmp.x, tmp.y, tmp.z);
    }
};

TC_INTERFACE(Texture);

TC_NAMESPACE_END

