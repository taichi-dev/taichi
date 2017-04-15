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
    Array2D <int64> max_dt_int_strength;
    Array2D <int64> max_dt_int_cfl;
    Array2D <int64> max_dt_int;
    Array2D <int64> update_propagation;
    Array2D<int> states;
    Array2D<int> updated;
    Array2D <Vector4> min_max_vel;
    Array2D <Vector4> min_max_vel_expanded;
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
        min_max_vel.initialize(res, Vector4(0));
        min_max_vel = Vector4(1e30f, 1e30f, -1e30f, -1e30f);
        min_max_vel_expanded.initialize(res, Vector4(0));
        max_dt_int_strength.initialize(res, 0);
        max_dt_int_cfl.initialize(res, 0);
        max_dt_int.initialize(res, 0);
        update_propagation.initialize(res, 1LL << 60);
    }

    void reset() {
        states = 0;
        max_dt_int.reset(1LL << 60);
    }

	bool has_particle(const Index2D &ind) {
		return particle_groups[ind.i * res[1] + ind.j].size() > 0;
	}

    void expand(bool expand_vel, bool expand_state) {
        Array2D<int> new_states;
        Array2D<Vector4> new_min_max_vel;
        new_min_max_vel.initialize(res, Vector4(1e30f, 1e30f, -1e30f, -1e30f));
        min_max_vel_expanded = Vector4(1e30f, 1e30f, -1e30f, -1e30f);
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
                if (states[ind])
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
                update(ind, 0, -1, new_min_max_vel, min_max_vel_expanded, new_states, states);
            }
            if (ind.j < states.get_height() - 1) {
                update(ind, 0, 1, new_min_max_vel, min_max_vel_expanded, new_states, states);
            }
        }
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
			for (auto &p : particle_groups[res[1] * ind.i + ind.j]) {
				int64 march_interval;
				int64 allowed_t_int_inc = (int64)(p->get_allowed_dt() / base_delta_t);
				if (allowed_t_int_inc <= 0) {
					P(allowed_t_int_inc);
					allowed_t_int_inc = 1;
				}
				march_interval = get_largest_pot(allowed_t_int_inc);
				p->march_interval = march_interval;
				Vector2i low_res_pos(int(p->pos.x / grid_block_size), int(p->pos.y / grid_block_size));
				max_dt_int_strength[low_res_pos] = std::min(max_dt_int_strength[low_res_pos],
															march_interval);
				auto &tmp = min_max_vel[low_res_pos.x][low_res_pos.y];
				tmp[0] = std::min(tmp[0], p->v.x);
				tmp[1] = std::min(tmp[1], p->v.y);
				tmp[2] = std::max(tmp[2], p->v.x);
				tmp[3] = std::max(tmp[3], p->v.y);
			}
		}
        // Expand velocity
        expand(true, false);

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
            real distance2boundary = std::max(
                    levelset->sample(Vector2(ind.get_pos() * real(grid_block_size)), t) - real(grid_block_size) * 0.75f,
                    0.5f);
            int64 boundary_limit = int64(cfl * distance2boundary / block_absolute_vel / base_delta_t);
            cfl_limit = std::min(cfl_limit, boundary_limit);
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
};

TC_NAMESPACE_END

