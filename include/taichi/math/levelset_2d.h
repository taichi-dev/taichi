#pragma once

#include <limits>
#include <taichi/math/array_2d.h>


TC_NAMESPACE_BEGIN

class LevelSet2D : public Array2D<float> {
public:
    float friction = 1.0f;
    LevelSet2D() : LevelSet2D(0, 0) {}
    LevelSet2D(int width, int height, Vector2 offset=Vector2(0.5f, 0.5f)) {
        initialize(width, height, offset);
    }
    void initialize(int width, int height, Vector2 offset) {
        Array2D<float>::initialize(width, height, 1, offset);
    }
    void add_sphere(Vector2 center, float radius, bool inside_out = false);

    void add_polygon(std::vector<Vector2> polygon, bool inside_out = false);

    Vector2 get_gradient(const Vector2 &pos) const; // Note this is not normalized!

    Vector2 get_normalized_gradient(const Vector2 &pos) const;

    static float fraction_outside(float phi_a, float phi_b) {
        return 1.0f - fraction_inside(phi_a, phi_b);
    }

    static float fraction_inside(float phi_a, float phi_b) {
        if (phi_a < 0 && phi_b < 0)
            return 1;
        if (phi_a < 0 && phi_b >= 0)
            return phi_a / (phi_a - phi_b);
        if (phi_a >= 0 && phi_b < 0)
            return phi_b / (phi_b - phi_a);
        else
            return 0;
    }

    Array2D<float> rasterize(int width, int height);
};


TC_NAMESPACE_END

