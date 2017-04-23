/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "levelset_3d.h"

TC_NAMESPACE_BEGIN

const real LevelSet3D::INF = 1e7;

void LevelSet3D::add_sphere(Vector3 center, real radius, bool inside_out) {
    for (auto &ind : get_region()) {
        Vector3 sample = ind.get_pos();
        real dist = (inside_out ? -1 : 1) * (length(center - sample) - radius);
        set(ind, std::min(Array3D::get(ind), dist));
    }
}

void LevelSet3D::add_plane(real a, real b, real c, real d) {
//    auto intersect = [&](real a, real b) {
//        real r = 2;
//        Vector2 u(std::max(r + a, 0.f), std::max(r + b, 0.f));
//        return std::min(-r, std::max(a, b)) + length(u);
//    };
    real coeff = 1.0f / sqrt(a * a + b * b + c * c);
    for (auto &ind : get_region()) {
        Vector3 sample = ind.get_pos();
        real dist = (glm::dot(sample, Vector3(a, b, c)) + d) * coeff;
//        if (Array3D::get(ind) == INF)
//            set(ind, dist);
//        else
//          set(ind, intersect(Array3D::get(ind), dist));
        set(ind, std::min(Array3D::get(ind), dist));
    }
}

void LevelSet3D::global_increase(real delta) {
    for (auto &ind : get_region()) {
        set(ind, Array3D::get(ind) + delta);
    }
}

Vector3 LevelSet3D::get_gradient(const Vector3 &pos) const {
    assert_info(inside(pos), "LevelSet Gradient Query out of Bound! ("
                             + std::to_string(pos.x) + ", "
                             + std::to_string(pos.y) + ", "
                             + std::to_string(pos.z) + ")");
    real x = pos.x, y = pos.y, z = pos.z;
    x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
    z = clamp(z - storage_offset.z, 0.f, depth - 1.f - eps);
    const int x_i = clamp(int(x), 0, width - 2);
    const int y_i = clamp(int(y), 0, height - 2);
    const int z_i = clamp(int(z), 0, depth - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real z_r = z - z_i;
    const real gx = lerp(y_r,
                         lerp(z_r, Array3D::get(x_i + 1, y_i, z_i) - Array3D::get(x_i, y_i, z_i),
                              Array3D::get(x_i + 1, y_i, z_i + 1) - Array3D::get(x_i, y_i, z_i + 1)),
                         lerp(z_r, Array3D::get(x_i + 1, y_i + 1, z_i) - Array3D::get(x_i, y_i + 1, z_i),
                              Array3D::get(x_i + 1, y_i + 1, z_i + 1) - Array3D::get(x_i, y_i + 1, z_i + 1)));
    const real gy = lerp(z_r,
                         lerp(x_r, Array3D::get(x_i, y_i + 1, z_i) - Array3D::get(x_i, y_i, z_i),
                              Array3D::get(x_i + 1, y_i + 1, z_i) - Array3D::get(x_i + 1, y_i, z_i)),
                         lerp(x_r, Array3D::get(x_i, y_i + 1, z_i + 1) - Array3D::get(x_i, y_i, z_i + 1),
                              Array3D::get(x_i + 1, y_i + 1, z_i + 1) - Array3D::get(x_i + 1, y_i, z_i + 1)));
    const real gz = lerp(x_r,
                         lerp(y_r, Array3D::get(x_i, y_i, z_i + 1) - Array3D::get(x_i, y_i, z_i),
                              Array3D::get(x_i, y_i + 1, z_i + 1) - Array3D::get(x_i, y_i + 1, z_i)),
                         lerp(y_r, Array3D::get(x_i + 1, y_i, z_i + 1) - Array3D::get(x_i + 1, y_i, z_i),
                              Array3D::get(x_i + 1, y_i + 1, z_i + 1) - Array3D::get(x_i + 1, y_i + 1, z_i)));
    return Vector3(gx, gy, gz);
}

Vector3 LevelSet3D::get_normalized_gradient(const Vector3 &pos) const {
    Vector3 gradient = get_gradient(pos);
    if (length(gradient) < 1e-10f)
        return Vector3(1, 0, 0);
    else
        return normalize(gradient);
}

real LevelSet3D::get(const Vector3 &pos) const {
    assert_info(inside(pos), "LevelSet Query out of Bound! ("
                             + std::to_string(pos.x) + ", "
                             + std::to_string(pos.y) + ", "
                             + std::to_string(pos.z) + ")");
    real x = pos.x, y = pos.y, z = pos.z;
    x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
    z = clamp(z - storage_offset.z, 0.f, depth - 1.f - eps);
    const int x_i = clamp(int(x), 0, width - 2);
    const int y_i = clamp(int(y), 0, height - 2);
    const int z_i = clamp(int(z), 0, depth - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real z_r = z - z_i;
    return lerp(x_r,
                lerp(y_r,
                     lerp(z_r, Array3D::get(x_i, y_i, z_i), Array3D::get(x_i, y_i, z_i + 1)),
                     lerp(z_r, Array3D::get(x_i, y_i + 1, z_i), Array3D::get(x_i, y_i + 1, z_i + 1))),
                lerp(y_r,
                     lerp(z_r, Array3D::get(x_i + 1, y_i, z_i), Array3D::get(x_i + 1, y_i, z_i + 1)),
                     lerp(z_r, Array3D::get(x_i + 1, y_i + 1, z_i), Array3D::get(x_i + 1, y_i + 1, z_i + 1))));
}


Array3D<real> LevelSet3D::rasterize(int width, int height, int depth) {
    for (auto &p : (*this)) {
        if (std::isnan(p)) {
            printf("Warning: nan in levelset.");
        }
    }
    Array3D<real> out(width, height, depth);
    Vector3 actual_size;
    if (storage_offset == Vector3(0.0f, 0.0f, 0.0f)) {
        actual_size = Vector3(this->width - 1, this->height - 1, this->depth - 1);
    } else {
        actual_size = Vector3(this->width, this->height, this->depth);
    }

    Vector3 scale_factor = actual_size / Vector3(width, height, depth);

    for (auto &ind : Region3D(0, width, 0, height, 0, depth, Vector3(0.5f, 0.5f, 0.5f))) {
        Vector3 p = scale_factor * ind.get_pos();
        out[ind] = sample(p);
        if (std::isnan(out[ind])) {
            out[ind] = std::numeric_limits<real>::infinity();
        }
    }
    return out;
}

TC_NAMESPACE_END
