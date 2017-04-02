/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "mpm_utils.h"
#include <stb_image.h>
#include <algorithm>
#include <atomic>
#include <taichi/math/array_2d.h>
#include <taichi/math/array_1d.h>
#include <taichi/math/levelset_2d.h>
#include "mpm_particle.h"

TC_NAMESPACE_BEGIN

typedef MPMParticle Particle;

class Grid {
public:
    Array2D<Vector2> velocity;
    Array2D<Vector2> force_or_acc;
    Array2D<Vector2> velocity_backup;
    Array2D<vec4> boundary_normal;
    Array2D<real> mass;
    Array2D<int> states;
    Vector2i res;

    Vector2i low_res;

    void initialize(const Vector2i &res) {
        this->res = res;
        velocity.initialize(res);
        force_or_acc.initialize(res);
        boundary_normal.initialize(res);
        mass.initialize(res);
        low_res.x = (res.x + grid_block_size - 1) / grid_block_size;
        low_res.y = (res.y + grid_block_size - 1) / grid_block_size;
        states.initialize(low_res);
    }

    void expand() {
        Array2D<int> new_states;
        new_states = states;
        // Expand x
        for (auto &ind: states.get_region()) {
            if (states[ind]) {
                new_states[ind] = 1;
                if (ind.i > 0) {
                    new_states[ind.neighbour(-1, 0)] = 1;
                }
                if (ind.i < states.get_width() - 1) {
                    new_states[ind.neighbour(1, 0)] = 1;
                }
            }
        }
        // Expand y
        for (auto &ind: states.get_region()) {
            if (new_states[ind]) {
                states[ind] = 1;
                if (ind.j > 0) {
                    states[ind.neighbour(0, -1)] = 1;
                }
                if (ind.j < states.get_height() - 1) {
                    states[ind.neighbour(0, 1)] = 1;
                }
            }
        }
    }

    int get_num_active_grids() {
        return states.abs_sum();
    }

    void backup_velocity() {
        velocity_backup = velocity;
    }

    void reset() {
        velocity = Vector2(0.0f);
        force_or_acc = Vector2(0.0f);
        mass = 0.0f;
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

    void apply_boundary_conditions(const LevelSet2D &levelset, real delta_t);

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

