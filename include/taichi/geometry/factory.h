#pragma once

#include <taichi/common/meta.h>

#include <taichi/math/linalg.h>
#include <taichi/math/array_2d.h>

#include <taichi/geometry/primitives.h>

TC_NAMESPACE_BEGIN

typedef std::function<Vector3(Vector2)> Function23;
typedef std::function<Vector2(Vector2)> Function22;
class Mesh3D {
public:
    // norm and uv can be null
    static std::vector<Triangle> generate(const Vector2i res,
        const Function23 *surf, const Function23 *norm, const Function22 *uv,
        bool smooth_normal);
};


TC_NAMESPACE_END
