#pragma once

#include <taichi/math/linalg.h>
#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

// I guess looking up a pointer can be more costy than returning 1 or 2 more floats, thus let's just
// treat Textued2D as special case of Texture3D.
// Maybe I'll do it better when I have time.

class Texture {
public:
    virtual void initialize(const Config &config) {}
    virtual Vector3 sample(const Vector2 &coord) const {return sample(Vector3(coord.x, coord.y, 0.0f));}
    virtual Vector3 sample(const Vector3 &coord) const {error("no impl"); return Vector3(0.0f);}
};

TC_INTERFACE(Texture);

TC_NAMESPACE_END

