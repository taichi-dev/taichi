#pragma once

#include <taichi/common/meta.h>

TC_NAMESPACE_BEGIN

    class Mesh3D {
    public:
        typedef std::function<Vector3(real, real)> SurfaceGenerator;

        static void generate(const SurfaceGenerator &func, Vector2i res) {
            for (int i = 0; i < res[0]; i++) {
                for (int j = 0; j < res[1]; j++) {
                    P(func(i, j))
                }
            }
        }
    };


TC_NAMESPACE_END
