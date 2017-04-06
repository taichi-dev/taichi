/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm_grid.h"

TC_NAMESPACE_BEGIN

long long MPMParticle::instance_count = 0;

void Grid::apply_boundary_conditions(const DynamicLevelSet2D & levelset, real delta_t, real t) {
    if (levelset.levelset0->get_width() > 0) {
        for (auto &ind : boundary_normal.get_region()) {
            Vector2 pos = Vector2(ind.i + 0.5f, ind.j + 0.5f);
            Vector2 v = velocity[ind] + force_or_acc[ind] * delta_t - levelset.get_temporal_derivative(pos, t) * levelset.get_spatial_gradient(pos, t);
            Vector2 n = levelset.get_spatial_gradient(pos, t);
            real phi = levelset.sample(pos, t);
            if (phi > 1) continue;
            else if (phi > 0) { // 0~1
                real pressure = std::max(-glm::dot(v, n), 0.0f);
                real mu = levelset.levelset0->friction;
                if (mu < 0) { // sticky
                    v = Vector2(0.0f);
                }
                else {
                    Vector2 t = Vector2(-n.y, n.x);
                    real friction = -clamp(glm::dot(t, v), -mu * pressure, mu * pressure);
                    v = v + n * pressure + t * friction;
                }
            }
            else if (phi <= 0) {
                v = Vector2(0.0f);
            }
            v += levelset.get_temporal_derivative(pos, t) * levelset.get_spatial_gradient(pos, t);
            force_or_acc[ind] = (v - velocity[ind]) / delta_t;
        }
    }
}

TC_NAMESPACE_END
