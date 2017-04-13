/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "mpm_utils.h"
#include <taichi/math/array_2d.h>
#include <taichi/math/dynamic_levelset_2d.h>

TC_NAMESPACE_BEGIN

class MPMScheduler {
public:
    Array2D <int64> max_dt_int_strength;
    Array2D <int64> max_dt_int_cfl;
    Array2D <int64> max_dt_int;
    Array2D<int> particle_count;
    Array2D<int> states;
    Array2D <Vector4> min_max_vel;
    Vector2i res;

    void initialize(const Vector2i &sim_res) {
        res.x = (sim_res.x + grid_block_size - 1) / grid_block_size;
        res.y = (sim_res.y + grid_block_size - 1) / grid_block_size;

        states.initialize(res, 0);
        particle_count.initialize(res, 0);
        min_max_vel.initialize(res, Vector4(0));
        max_dt_int_strength.initialize(res, 0);
        max_dt_int_cfl.initialize(res, 0);
        max_dt_int.initialize(res, 0);
    }

    void reset() {
        states = 0;
        particle_count = 0;
        min_max_vel = Vector4(1e30f, 1e30f, -1e30f, -1e30f);
        max_dt_int_strength.reset(1LL << 60);
        max_dt_int_cfl.reset(1LL << 60);
        max_dt_int.reset(1LL << 60);
    }

    void expand(bool expand_vel, bool expand_state) {
        Array2D<int> new_states;
        Array2D<Vector4> new_min_max_vel;
        new_min_max_vel.initialize(res, Vector4(1e30f, 1e30f, -1e30f, -1e30f));
        new_states.initialize(res, 0);

        auto update = [&](const Index2D ind, int dx, int dy,
                          const Array2D<Vector4> &min_max_vel, Array2D<Vector4> &new_min_max_vel,
                          const Array2D<int> &states, Array2D<int> &new_states) -> void {
            if (expand_vel) {
                auto &tmp = new_min_max_vel[ind.neighbour(dx, dy)];
                tmp[0] = std::min(tmp[0], min_max_vel[ind][0]);
                tmp[1] = std::min(tmp[1], min_max_vel[ind][1]);
                tmp[2] = std::max(tmp[2], min_max_vel[ind][2]);
                tmp[3] = std::max(tmp[3], min_max_vel[ind][3]);
            }
            if (expand_state) {
                if (states[ind.neighbour(dx, dy)])
                    new_states[ind.neighbour(dx, dy)] = 1;
            }
        };

        // Expand x
        for (auto &ind : states.get_region()) {
            update(ind, 0, 0, min_max_vel, new_min_max_vel, states, new_states);
            if (ind.i > 0) {
                update(ind, -1, 0, min_max_vel, new_min_max_vel, states, new_states);
            }
            if (ind.i < states.get_width() - 1) {
                update(ind, 1, 0, min_max_vel, new_min_max_vel, states, new_states);
            }
        }
        // Expand y
        for (auto &ind : states.get_region()) {
            update(ind, 0, 0, new_min_max_vel, min_max_vel, new_states, states);
            if (ind.j > 0) {
                update(ind, 0, -1, new_min_max_vel, min_max_vel, new_states, states);
            }
            if (ind.j < states.get_height() - 1) {
                update(ind, 0, 1, new_min_max_vel, min_max_vel, new_states, states);
            }
        }
    }

    void update() {

    }

    int get_num_active_grids() {
        return states.abs_sum();
    }
};

TC_NAMESPACE_END

