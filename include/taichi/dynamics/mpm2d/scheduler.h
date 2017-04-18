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
    Array2D<int64> max_dt_int_strength;
    Array2D<int64> max_dt_int_cfl;
    Array2D<int64> max_dt_int;
    Array2D<int> states;
    Array2D<int> updated;
    Array2D<Vector4> min_max_vel;
    Array2D<Vector4> min_max_vel_expanded;
    std::vector<std::vector<Particle *>> particle_groups;
    Vector2i res;
    Vector2i sim_res;
    std::vector<Particle *> active_particles;
    std::vector<Vector2i> active_grid_points;
    DynamicLevelSet2D *levelset;
    real base_delta_t;
    real cfl;

    void initialize(const Vector2i &sim_res, real base_delta_t, real cfl, DynamicLevelSet2D *levelset) {
        this->sim_res = sim_res;
        res.x = (sim_res.x + grid_block_size - 1) / grid_block_size;
        res.y = (sim_res.y + grid_block_size - 1) / grid_block_size;

        this->base_delta_t = base_delta_t;
        this->levelset = levelset;
        this->cfl = cfl;

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
        max_dt_int.initialize(res, 0);
    }

    void reset() {
        states = 0;
        max_dt_int.initialize(res, 1LL << 60LL);
    }

    bool has_particle(const Index2D &ind) {
        return has_particle(Vector2i(ind.i, ind.j));
    }

    bool has_particle(const Vector2i &ind) {
        return particle_groups[ind.x * res[1] + ind.y].size() > 0;
    }

    void expand(bool expand_vel, bool expand_state, bool expand_propagation) {
        Array2D<int> new_states;
        Array2D<int> old_states;
        if (expand_state) {
            old_states = states;
        }
        Array2D<Vector4> new_min_max_vel;
        new_min_max_vel.initialize(res, Vector4(1e30f, 1e30f, -1e30f, -1e30f));
        min_max_vel_expanded = Vector4(1e30f, 1e30f, -1e30f, -1e30f);
        new_states.initialize(res, 0);
        Array2D<int64> new_max_dt_int;

        auto update = [&](const Index2D ind, int dx, int dy,
                          const Array2D<Vector4> &min_max_vel, Array2D<Vector4> &new_min_max_vel,
                          const Array2D<int> &states, Array2D<int> &new_states,
                          const Array2D<int64> &max_dt_int,
                          Array2D<int64> &new_max_dt_int) -> void {
            if (expand_vel) {
                auto &tmp = new_min_max_vel[ind.neighbour(dx, dy)];
                tmp[0] = std::min(tmp[0], min_max_vel[ind][0]);
                tmp[1] = std::min(tmp[1], min_max_vel[ind][1]);
                tmp[2] = std::max(tmp[2], min_max_vel[ind][2]);
                tmp[3] = std::max(tmp[3], min_max_vel[ind][3]);
            }
            if (expand_state) {
                if (states[ind])
                    new_states[ind.neighbour(dx, dy)] = 1;
            }
        };

        // Expand x
        for (auto &ind : states.get_region()) {
            for (int dx = -1; dx <= 1; dx++) {
                if (0 <= ind.i + dx && ind.i + dx < states.get_width())
                    update(ind, dx, 0, min_max_vel, new_min_max_vel, states, new_states, max_dt_int,
                           new_max_dt_int);
            }
        }
        // Expand y
        for (auto &ind : states.get_region()) {
            for (int dy = -1; dy <= 1; dy++) {
                if (0 <= ind.j + dy && ind.j + dy < states.get_height())
                    update(ind, 0, dy, new_min_max_vel, min_max_vel_expanded, new_states, states,
                           new_max_dt_int, max_dt_int);
            }
        }
        if (expand_state) {
            states += old_states;
        } // 1: buffer, 2: updating
    }

    void update() {
        // Use <= here since grid_res = sim_res + 1
        active_particles.clear();
        active_grid_points.clear();
        for (int i = 0; i <= sim_res[0]; i++) {
            for (int j = 0; j <= sim_res[1]; j++) {
                if (states[i / grid_block_size][j / grid_block_size] == 1) {
                    active_grid_points.push_back(Vector2i(i, j));
                }
            }

        }
        for (auto &ind : states.get_region()) {
            if (states[ind] != 0) {
                for (auto &p : particle_groups[res[1] * ind.i + ind.j]) {
                    active_particles.push_back(p);
                }
            }
        }
    }

    void update_particle_groups() {
        // Remove all updating particles, and then re-insert them
        for (auto &ind : states.get_region()) {
            if (states[ind] == 0) {
                continue;
            }
            particle_groups[res[1] * ind.i + ind.j].clear();
            updated[ind] = 1;
        }
        for (auto &p : active_particles) {
            insert_particle(p);
        }
    }

    void insert_particle(Particle *p) {
        int x = int(p->pos.x / grid_block_size);
        int y = int(p->pos.y / grid_block_size);
        if (states.inside(x, y)) {
            int index = res[1] * x + y;
            particle_groups[index].push_back(p);
            updated[x][y] = 1;
        }
    }

    void update_dt_limits(real t) {
        for (auto &ind : states.get_region()) {
            // Update those blocks needing an update
            if (!updated[ind]) {
                continue;
            }
            updated[ind] = 0;
            max_dt_int_strength[ind] = 1LL << 60;
            max_dt_int_cfl[ind] = 1LL << 60;
            min_max_vel[ind] = Vector4(1e30f, 1e30f, -1e30f, -1e30f);
            for (auto &p : particle_groups[res[1] * ind.i + ind.j]) {
                int64 march_interval;
                int64 allowed_t_int_inc = (int64)(p->get_allowed_dt() / base_delta_t);
                if (allowed_t_int_inc <= 0) {
                    P(allowed_t_int_inc);
                    allowed_t_int_inc = 1;
                }
                march_interval = get_largest_pot(allowed_t_int_inc);
                p->march_interval = march_interval;
                max_dt_int_strength[ind] = std::min(max_dt_int_strength[ind],
                                                    march_interval);
                auto &tmp = min_max_vel[ind];
                tmp[0] = std::min(tmp[0], p->v.x);
                tmp[1] = std::min(tmp[1], p->v.y);
                tmp[2] = std::max(tmp[2], p->v.x);
                tmp[3] = std::max(tmp[3], p->v.y);
            }
        }
        // Expand velocity
        expand(true, false, false);

        for (auto &ind : min_max_vel.get_region()) {
            real block_vel = std::max(
                    min_max_vel_expanded[ind][2] - min_max_vel_expanded[ind][0],
                    min_max_vel_expanded[ind][3] - min_max_vel_expanded[ind][1]
            ) + 1e-7f;
            if (block_vel < 0) {
                // Blocks with no particles
                continue;
            }
            int64 cfl_limit = int64(cfl / block_vel / base_delta_t);
            if (cfl_limit <= 0) {
                P(cfl_limit);
                cfl_limit = 1;
            }
            real block_absolute_vel = 1e-7f;
            for (int i = 0; i < 4; i++) {
                block_absolute_vel = std::max(block_absolute_vel, std::abs(min_max_vel_expanded[ind][i]));
            }
            real last_distance = levelset->sample(Vector2(ind.get_pos() * real(grid_block_size)), t);
            if (last_distance < LevelSet2D::INF) {
                real distance2boundary = std::max(last_distance - real(grid_block_size) * 0.75f, 0.5f);
                int64 boundary_limit = int64(cfl * distance2boundary / block_absolute_vel / base_delta_t);
                cfl_limit = std::min(cfl_limit, boundary_limit);
            }
            max_dt_int_cfl[ind] = get_largest_pot(cfl_limit);
        }
    }

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

    void visualize(const Vector4 &debug_input, Array2D<Vector4> &debug_blocks) const {
        int64 minimum = int64(debug_input[0]);
        if (minimum == 0) {
            for (auto &ind : max_dt_int_cfl.get_region()) {
                minimum = std::min(minimum, max_dt_int[ind]);
            }
        }
        minimum = std::max(minimum, 1LL);
        int grades = int(debug_input[1]);
        if (grades == 0) {
            grades = 10;
        }

        auto visualize = [](const Array2D<int64> step, int grades, int64 minimum) -> Array2D<real> {
            Array2D<real> output;
            output.initialize(step.get_width(), step.get_height());
            for (auto &ind : step.get_region()) {
                real r;
                r = 1.0f - std::log2(1.0f * (step[ind] / minimum)) / grades;
                output[ind] = clamp(r, 0.0f, 1.0f);
            }
            return output;
        };

        auto vis_strength = visualize(max_dt_int_strength, grades, minimum);
        auto vis_cfl = visualize(max_dt_int_cfl, grades, minimum);
        for (auto &ind : min_max_vel.get_region()) {
            debug_blocks[ind] = Vector4(vis_strength[ind], vis_cfl[ind], 0.0f, 1.0f);
        }
    }

    void print_limits() {
        for (int i = max_dt_int.get_height() - 1; i >= 0; i--) {
            for (int j = 0; j < max_dt_int.get_width(); j++) {
                // std::cout << scheduler.particle_groups[j * scheduler.res[1] + i].size() << " " << (int)scheduler.has_particle(Vector2i(j, i)) << "; ";
                printf(" %f", min_max_vel[j][i][0]);
            }
            printf("\n");
        }
        printf("\n");
        P(get_active_particles().size());
        for (int i = max_dt_int.get_height() - 1; i >= 0; i--) {
            for (int j = 0; j < max_dt_int.get_width(); j++) {
                if (max_dt_int[j][i] >= (1LL << 60)) {
                    printf("      #");
                } else {
                    printf("%6lld", max_dt_int_strength[j][i]);
                    if (states[j][i] == 1) {
                        printf("*");
                    } else {
                        printf(" ");
                    }
                }
            }
            printf("\n");
        }
        printf("\n");
        printf("cfl\n");
        for (int i = max_dt_int.get_height() - 1; i >= 0; i--) {
            for (int j = 0; j < max_dt_int.get_width(); j++) {
                if (max_dt_int[j][i] >= (1LL << 60)) {
                    printf("      #");
                } else {
                    printf("%6lld", max_dt_int_cfl[j][i]);
                    if (states[j][i] == 1) {
                        printf("*");
                    } else {
                        printf(" ");
                    }
                }
            }
            printf("\n");
        }
        printf("\n");
    }

    void update_particle_states() {
        for (auto &p : get_active_particles()) {
            p->color = Vector3(1.0f);
            Vector2i low_res_pos(int(p->pos.x / grid_block_size), int(p->pos.y / grid_block_size));
            p->march_interval = max_dt_int[low_res_pos];
        }
    }

    void reset_particle_states() {
        for (auto &p : get_active_particles()) {
            Vector2i low_res_pos(int(p->pos.x / grid_block_size), int(p->pos.y / grid_block_size));
            if (states[low_res_pos] == 0) {
                p->state = MPMParticle::INACTIVE;
                p->color = Vector3(0.3f);
                continue;
            }
        }
    }
};

TC_NAMESPACE_END

