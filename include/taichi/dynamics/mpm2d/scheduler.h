/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "mpm_utils.h"
#include "mpm_particle.h"
#include <taichi/math/array_2d.h>
#include <taichi/math/dynamic_levelset_2d.h>

TC_NAMESPACE_BEGIN

class MPMScheduler {
public:
    typedef MPMParticle Particle;

    template <typename T> using Array = Array2D<T>;

    Array<int64> max_dt_int_strength;
    Array<int64> max_dt_int_cfl;
    Array<int64> max_dt_int;
    Array<int> states;
    Array<int> updated;
    Array<Vector4> min_max_vel;
    Array<Vector4> min_max_vel_expanded;
    std::vector<std::vector<Particle *>> particle_groups;
    Vector2i res;
    Vector2i sim_res;
    std::vector<Particle *> active_particles;
    std::vector<Vector2i> active_grid_points;
    DynamicLevelSet2D *levelset;
    real base_delta_t;
    real cfl, strength_dt_mul;

    void initialize(const Vector2i &sim_res, real base_delta_t, real cfl, real strength_dt_mul,
                    DynamicLevelSet2D *levelset) {
        this->sim_res = sim_res;
        res.x = (sim_res.x + grid_block_size - 1) / grid_block_size;
        res.y = (sim_res.y + grid_block_size - 1) / grid_block_size;

        this->base_delta_t = base_delta_t;
        this->levelset = levelset;
        this->cfl = cfl;
        this->strength_dt_mul = strength_dt_mul;

        states.initialize(res, 0);
        updated.initialize(res, 1);
        particle_groups.resize(res[0] * res[1]);
        for (int i = 0; i < res[0] * res[1]; i++) {
            particle_groups[i] = std::vector<Particle *>();
        }
        min_max_vel.initialize(res, Vector4(0));
        min_max_vel = Vector4(1e30f, 1e30f, -1e30f, -1e30f);
        min_max_vel_expanded.initialize(res, Vector4(0));
        max_dt_int_strength.initialize(res, 0);
        max_dt_int_cfl.initialize(res, 0);
        max_dt_int.initialize(res, 1);
    }

    void reset() {
        states = 0;
    }

    bool has_particle(const Index2D &ind) {
        return has_particle(Vector2i(ind.i, ind.j));
    }

    bool has_particle(const Vector2i &ind) {
        return particle_groups[ind.x * res[1] + ind.y].size() > 0;
    }

    void expand(bool expand_vel, bool expand_state);

    void update();

    int64 update_max_dt_int(int64 t_int);

    void set_time(int64 t_int) {
        for (auto &ind : states.get_region()) {
            if (t_int % max_dt_int[ind] == 0) {
                states[ind] = 1;
            }
        }
    }

    void update_particle_groups();

    void insert_particle(Particle *p);

    void update_dt_limits(real t);

    int get_num_active_grids() {
        int count = 0;
        for (auto &ind : states.get_region()) {
            count += int(states[ind] != 0);
        }
        return count;
    }

    const std::vector<Particle *> &get_active_particles() const {
        return active_particles;
    }

    const std::vector<Vector2i> &get_active_grid_points() const {
        return active_grid_points;
    }

    void visualize(const Vector4 &debug_input, Array<Vector4> &debug_blocks) const;

    void print_limits();

    void print_max_dt_int();

    void update_particle_states();

    void reset_particle_states();

    void enforce_smoothness(int64 t_int_increment);
};

TC_NAMESPACE_END

