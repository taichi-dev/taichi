/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm_grid.h"

TC_NAMESPACE_BEGIN

long long MPMParticle::instance_count = 0;

void Grid::apply_boundary_conditions(const DynamicLevelSet2D &levelset, real delta_t, real t) {
    for (auto &ind : scheduler->get_active_grid_points()) {
        Vector2 pos = Vector2(ind[0] + 0.5f, ind[1] + 0.5f);
        real phi = levelset.sample(pos, t);
        if (phi > 1) continue;
        Vector2 n = levelset.get_spatial_gradient(pos, t);
        Vector2 boundary_velocity = levelset.get_temporal_derivative(pos, t) * n;
        Vector2 v = velocity[ind] - boundary_velocity;
        if (phi > 0) { // 0~1
            real pressure = std::max(-glm::dot(v, n), 0.0f);
            real mu = levelset.levelset0->friction;
            if (mu < 0) { // sticky
                v = Vector2(0.0f);
            } else {
                Vector2 t = v - n * glm::dot(v, n);
                if (length(t) > 1e-6f) {
                    t = normalize(t);
                }
                real friction = -clamp(glm::dot(t, v), -mu * pressure, mu * pressure);
                v = v + n * pressure + t * friction;
            }
        } else if (phi <= 0) {
            v = Vector2(0.0f);
        }
        v += boundary_velocity;
        velocity[ind] = v;
    }
}

TC_NAMESPACE_END
