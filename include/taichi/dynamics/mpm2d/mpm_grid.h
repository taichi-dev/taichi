/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "mpm_utils.h"
#include <algorithm>
#include <atomic>
#include <taichi/math/array_2d.h>
#include <taichi/math/array_1d.h>
#include <taichi/math/levelset_2d.h>
#include <taichi/math/dynamic_levelset_2d.h>
#include "mpm_particle.h"
#include "mpm_scheduler.h"

TC_NAMESPACE_BEGIN

typedef MPMParticle Particle;

class Grid {
public:
    Array2D<Vector2> velocity;
    Array2D<Vector2> force_or_acc;
    Array2D<Vector2> velocity_backup;
    Array2D<Vector4> boundary_normal;
    Array2D<real> mass;
    Vector2i res;
    MPMScheduler *scheduler;

    void initialize(const Vector2i &sim_res, MPMScheduler *scheduler) {
        this->res = sim_res + Vector2i(1);
        velocity.initialize(res, Vector2(0), Vector2(0));
        force_or_acc.initialize(res, Vector2(0), Vector2(0));
        boundary_normal.initialize(res, Vector4(0), Vector2(0));
        mass.initialize(res, 0.0f, Vector2(0));
        this->scheduler = scheduler;
    }

    void reset() {
        velocity = Vector2(0.0f);
        force_or_acc = Vector2(0.0f);
        mass = 0.0f;
    }

    void backup_velocity() {
        velocity_backup = velocity;
    }

    void normalize_velocity() {
        for (auto &ind : velocity.get_region()) {
            if (mass[ind] > 0) { // Do not use EPS here!!
                velocity[ind] /= mass[ind];
            } else {
                velocity[ind] = Vector2(0, 0);
            }
            CV(velocity[ind]);
        }
    }

    void normalize_acceleration() {
        for (auto &ind : force_or_acc.get_region()) {
            if (mass[ind] > 0) { // Do not use EPS here!!
                force_or_acc[ind] /= mass[ind];
            } else {
                force_or_acc[ind] = Vector2(0, 0);
            }
            CV(force_or_acc[ind]);
        }
    }

    void apply_external_force(Vector2 acc) {
        for (auto &ind : mass.get_region()) {
            if (mass[ind] > 0) // Do not use EPS here!!
                force_or_acc[ind] += acc * mass[ind];
        }
    }

    void apply_boundary_conditions(const DynamicLevelSet2D &levelset, real delta_t, real t);

    void check_velocity() {
        for (int i = 0; i < res[0]; i++) {
            for (int j = 0; j < res[1]; j++) {
                if (!is_normal(velocity[i][j])) {
                    printf("Grid Velocity Check Fail!\n");
                    Pp(i);
                    Pp(j);
                    Pp(velocity[i][j]);
                    assert(false);
                }
            }
        }
    }
};


TC_NAMESPACE_END

