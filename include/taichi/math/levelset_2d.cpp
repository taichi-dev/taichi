/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "levelset_2d.h"

TC_NAMESPACE_BEGIN


void LevelSet2D::add_sphere(Vector2 center, real radius, bool inside_out) {
    for (auto &ind : get_region()) {
        Vector2 sample = ind.get_pos();
        real dist = (inside_out ? -1 : 1) * (length(center - sample) - radius);
        set(ind, std::min(get(ind), dist));
    }
}

void LevelSet2D::add_polygon(std::vector<Vector2> polygon, bool inside_out)
{
    for (auto &ind : get_region()) {
        Vector2 p = ind.get_pos();
        real dist = ((inside_polygon(p, polygon) ^ inside_out) ? -1 : 1) * (nearest_distance(p, polygon));
        set(ind, std::min(get(ind), dist));
    }
}

Vector2 LevelSet2D::get_gradient(const Vector2 &pos) const
{
    assert_info(inside(pos), "LevelSet Gradient Query out of Bound! (" + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ")");
    real x = pos.x, y = pos.y;
    x = clamp(x - storage_offset.x, 0.f, width - 1.f - eps);
    y = clamp(y - storage_offset.y, 0.f, height - 1.f - eps);
    const int x_i = clamp(int(x), 0, width - 2);
    const int y_i = clamp(int(y), 0, height - 2);
    const real x_r = x - x_i;
    const real y_r = y - y_i;
    const real gx = lerp(y_r, get(x_i + 1, y_i) - get(x_i, y_i), get(x_i + 1, y_i + 1) - get(x_i, y_i + 1));
    const real gy = lerp(x_r, get(x_i, y_i + 1) - get(x_i, y_i), get(x_i + 1, y_i + 1) - get(x_i + 1, y_i));
    return Vector2(gx, gy);
}

Vector2 LevelSet2D::get_normalized_gradient(const Vector2 &pos) const
{
    Vector2 gradient = get_gradient(pos);
    if (length(gradient) < 1e-10f)
        return Vector2(1, 0);
    else
        return normalize(gradient);
}

Array2D<real> LevelSet2D::rasterize(int width, int height) {
    for (auto &p : (*this)) {
        if (std::isnan(p)) {
            printf("Warning: nan in levelset.");
        }
    }
    Array2D<real> out(width, height);
    Vector2 actual_size;
    if (storage_offset == Vector2(0.0f, 0.0f)) {
        actual_size = Vector2(this->width - 1, this->height - 1);
    }
    else {
        actual_size = Vector2(this->width, this->height);
    }

    Vector2 scale_factor = actual_size / Vector2(width, height);

    for (auto &ind : Region2D(0, width, 0, height, Vector2(0.5f, 0.5f))) {
        Vector2 p = scale_factor * ind.get_pos();
        out[ind] = sample(p);
        if (std::isnan(out[ind])) {
            out[ind] = std::numeric_limits<real>::infinity();
        }
    }
    return out;
}

TC_NAMESPACE_END
