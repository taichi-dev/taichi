#pragma once

#include <taichi/common/meta.h>

#include <taichi/math/linalg.h>
#include <taichi/math/array_2d.h>

#include <taichi/geometry/primitives.h>

TC_NAMESPACE_BEGIN

    class Mesh3D {
    public:
        typedef std::function<Vector3(real, real)> SurfaceGenerator;

        static std::vector<Triangle> generate(const SurfaceGenerator &func, Vector2i res);
    };


TC_NAMESPACE_END
