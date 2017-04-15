/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm.h"
#include <taichi/common/asset_manager.h>

TC_NAMESPACE_BEGIN

long long kernel_calc_counter = 0;

void MPM::initialize(const Config &config_) {
    auto config = Config(config_);
    this->async = config.get("async", false);
    res = config.get_vec2i("res");
    this->cfl = config.get("cfl", 1.0f);
    this->apic = config.get("apic", true);
    this->h = config.get_real("delta_x");
    this->kill_at_boundary = config.get("kill_at_boundary", true);
    grid.initialize(res);
    t = 0.0f;
    t_int = 0;
    requested_t = 0.0f;
    if (!apic) {
        flip_alpha = config.get_real("flip_alpha");
        flip_alpha_stride = config.get_real("flip_alpha_stride");
    } else {
        flip_alpha = 1.0f;
        flip_alpha_stride = 1.0f;
    }
    gravity = config.get("gravity", Vector2(0, -10));
    base_delta_t = config.get("base_delta_t", 1e-6f);
    particle_collision = config.get("particle_collision", true);
    position_noise = config.get("position_noise", 0.5f);
    if (async) {
        maximum_delta_t = config.get("maximum_delta_t", 1e-1);
    } else {
        maximum_delta_t = base_delta_t;
    }
    material_levelset.initialize(res + Vector2i(1), Vector2(0));
    this->debug_input = config.get("debug_input", Vector4(0, 0, 0, 0));
    scheduler.initialize(res);
    debug_blocks.initialize(scheduler.res, Vector4(0), Vector2(0));
}

void MPM::substep() {
    real delta_t = base_delta_t;

    bool exist_updating_particle = false;

    grid.reset();
    int64 t_int_increment;

    if (async) {
        scheduler.reset();
        Array2D<int64> &max_dt_int_strength = scheduler.max_dt_int_strength;
        Array2D<int64> &max_dt_int_cfl = scheduler.max_dt_int_cfl;
        Array2D<int64> &max_dt_int = scheduler.max_dt_int;

        for (auto &p : particles) {
            int64 march_interval;
            int64 allowed_t_int_inc = (int64)(p->get_allowed_dt() / base_delta_t);
            if (allowed_t_int_inc <= 0) {
                P(allowed_t_int_inc);
                allowed_t_int_inc = 1;
            }
            march_interval = get_largest_pot(allowed_t_int_inc);
            p->march_interval = march_interval;
            Vector2i low_res_pos(int(p->pos.x / grid_block_size), int(p->pos.y / grid_block_size));
            scheduler.particle_count[low_res_pos] += 1;
            max_dt_int_strength[low_res_pos] = std::min(max_dt_int_strength[low_res_pos],
                                                        march_interval);
            auto &tmp = scheduler.min_max_vel[low_res_pos.x][low_res_pos.y];
            tmp[0] = std::min(tmp[0], p->v.x);
            tmp[1] = std::min(tmp[1], p->v.y);
            tmp[2] = std::max(tmp[2], p->v.x);
            tmp[3] = std::max(tmp[3], p->v.y);
        }
        // Expand velocity
        scheduler.expand(true, false);

        for (auto &ind : scheduler.min_max_vel.get_region()) {
            real block_vel = std::max(
                    scheduler.min_max_vel[ind][2] - scheduler.min_max_vel[ind][0],
                    scheduler.min_max_vel[ind][3] - scheduler.min_max_vel[ind][1]
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
                block_absolute_vel = std::max(block_absolute_vel, std::abs(scheduler.min_max_vel[ind][i]));
            }
            real distance2boundary = std::max(
                    levelset.sample(Vector2(ind.get_pos() * real(grid_block_size)), t) - real(grid_block_size) * 0.75f,
                    0.5f);
            int64 boundary_limit = int64(cfl * distance2boundary / block_absolute_vel / base_delta_t);
            cfl_limit = std::min(cfl_limit, boundary_limit);
            max_dt_int_cfl[ind] = get_largest_pot(cfl_limit);
        }

        t_int_increment = get_largest_pot(int64(maximum_delta_t / base_delta_t));

        for (auto &ind : scheduler.max_dt_int_strength.get_region()) {
            max_dt_int[ind] = std::min(max_dt_int_cfl[ind], max_dt_int_strength[ind]);
            // max_dt_int[ind] = std::min(max_dt_int[ind], scheduler.update_propagation[ind]);
            if (scheduler.particle_count[ind] == 0) {
                continue;
            }
            t_int_increment = std::min(t_int_increment, max_dt_int[ind]);
        }
        //scheduler.update_propagation.reset(1LL << 60);

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
        for (auto &ind : scheduler.min_max_vel.get_region()) {
            debug_blocks[ind] = Vector4(vis_strength[ind], vis_cfl[ind], 0.0f, 1.0f);
        }

		if (debug_input[2] > 0) {
			P(t_int_increment);
		}
        // t_int_increment is the biggest allowed dt.
        t_int_increment = t_int_increment - t_int % t_int_increment;

		if (debug_input[2] > 0) {
			P(t_int_increment);
			P(t_int);
		}

        t_int += t_int_increment; // final dt
        t = base_delta_t * t_int;

        int64 max_dt = 0, min_dt = 1LL << 60;

        for (auto &ind : scheduler.states.get_region()) {
            if (scheduler.particle_count[ind] == 0) {
                //continue;
            }
            max_dt = std::max(max_dt_int[ind], max_dt);
            min_dt = std::min(max_dt_int[ind], min_dt);
            if (t_int % max_dt_int[ind] == 0) {
                scheduler.states[ind] = 1;
                for (int dx = -1; dx < 2; dx++) {
                    for (int dy = -1; dy < 2; dy++) {
                        Vector2i pos(ind.i + dx, ind.j + dy);
                        if (scheduler.update_propagation.inside(pos.x, pos.y))
                            scheduler.update_propagation[pos] = std::min(scheduler.update_propagation[pos],
                                                                         max_dt_int[ind] * 2);
                    }
                }
            }
            // printf("t_int %lld max_dt_int %lld mod %lld %d %d\n", t_int, max_dt_int[ind], t_int % max_dt_int[ind], ind.i, ind.j);
        }
		if (debug_input[2] > 0) {
			printf("min_dt %lld max_dt %lld dynamic_range %lld\n", min_dt, max_dt, max_dt / min_dt);

			for (int i = scheduler.max_dt_int.get_height() - 1; i >= 0; i--) {
				for (int j = 0; j < scheduler.max_dt_int.get_width(); j++) {
					if (max_dt_int[j][i] >= (1LL << 60)) {
						printf("      #");
					} else {
						printf("%6lld", max_dt_int[j][i]);
						if (scheduler.states[j][i] == 1) {
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

        // TODO...
        exist_updating_particle = true;
        if (!exist_updating_particle) {
            return;
        }

        Array2D<int> old_grid_states = scheduler.states;
        // Expand state
        // P(scheduler.get_num_active_grids());
        scheduler.expand(false, true);
        // P(scheduler.get_num_active_grids());

        int active_particle_count = 0;
        int buffer_particle_count = 0;

        for (auto &p : particles) {
            Vector2i low_res_pos(int(p->pos.x / grid_block_size), int(p->pos.y / grid_block_size));
            if (scheduler.states[low_res_pos] == 0) {
                p->state = MPMParticle::INACTIVE;
                continue;
            }
            p->march_interval = max_dt_int[low_res_pos];
            if (old_grid_states[low_res_pos] == 1) {
                p->state = MPMParticle::UPDATING;
                active_particle_count += 1;
            } else {
                p->state = MPMParticle::BUFFER;
                buffer_particle_count += 1;
            }
        }

		if (debug_input[2] > 0) {
			P(active_particle_count);
			P(buffer_particle_count);
		}
    } else {
        // Sync
        t_int_increment = 1;
        for (auto &p : particles) {
            p->state = MPMParticle::UPDATING;
            p->march_interval = 1;
        }
        t_int += t_int_increment; // final dt
        t = base_delta_t * t_int;
    }

    for (auto &p : particles) {
        if (p->state != MPMParticle::INACTIVE)
            p->calculate_kernels();
    }
	scheduler.update(particles);

    rasterize();
    estimate_volume();
    grid.backup_velocity();
    grid.apply_external_force(gravity);
    apply_deformation_force();
    grid.normalize_acceleration();
    grid.apply_boundary_conditions(levelset, t_int_increment * base_delta_t, t);
    resample();
    std::vector<Particle *> new_particles;
    for (auto &p : particles) {
        bool killed = false;
        if (p->state == MPMParticle::UPDATING) {
            p->pos += (t_int - p->last_update) * delta_t * p->v;
            p->last_update = t_int;
            for (int i = 0; i < 2; i++) {
                if (p->pos[i] < 1.0f || p->pos[i] > res[i] - 1.0f) {
                    if (!kill_at_boundary) {
                        p->pos[i] = clamp(p->pos.x, 1.0f, res[i] - 1.0f);
                    } else {
                        killed = true;
                    }
                }
            }
        }
        if (!killed) {
            new_particles.push_back(p);
        }
    }
    particles.swap(new_particles);
    if (particle_collision)
        particle_collision_resolution();
}

void MPM::step(real delta_t) {
    if (delta_t < 0) {
        substep();
        requested_t = t;
    } else {
        requested_t += delta_t;
        while (t + base_delta_t < requested_t)
            substep();
    }
    compute_material_levelset();
    P(kernel_calc_counter);
}

void MPM::compute_material_levelset() {
    material_levelset.reset(std::numeric_limits<real>::infinity());
    for (auto &p : particles) {
        for (auto &ind : material_levelset.get_rasterization_region(p->pos, 3)) {
            Vector2 delta_pos = ind.get_pos() - p->pos;
            material_levelset[ind] = std::min(material_levelset[ind], length(delta_pos) - 0.8f);
        }
    }
    for (auto &ind : material_levelset.get_region()) {
        if (material_levelset[ind] < 0.5f) {
            if (levelset.sample(ind.get_pos(), t) < 0)
                material_levelset[ind] = -0.5f;
        }
    }
}

void MPM::particle_collision_resolution() {
    if (levelset.levelset0) {
        for (auto &p : particles) {
            if (p->state == MPMParticle::UPDATING)
                p->resolve_collision(levelset, t);
        }
    }
}

void MPM::estimate_volume() {
    for (auto &p : particles) {
        if (p->state != MPMParticle::INACTIVE && p->vol == -1.0f) {
            real rho = 0.0f;
            for (auto &ind : get_bounded_rasterization_region(p->pos)) {
                real weight = p->get_cache_w(ind);
                rho += grid.mass[ind] / h / h;
            }
            p->vol = p->mass / rho;
        }
    }
}

void MPM::add_particle(const Config &config) {
    error("no_impl");
}

void MPM::add_particle(std::shared_ptr<MPMParticle> p) {
    // WTH???
    p->mass = 1.0f / res[0] / res[0];
    p->pos += position_noise * Vector2(rand() - 0.5f, rand() - 0.5f);
    particles.push_back(p->duplicate());
}

void MPM::add_particle(EPParticle p) {
    add_particle(std::make_shared<EPParticle>(p));
}

void MPM::add_particle(DPParticle p) {
    add_particle(std::make_shared<DPParticle>(p));
}

std::vector<std::shared_ptr<MPMParticle>> MPM::get_particles() {
	std::vector<std::shared_ptr<MPMParticle>> particles;
	for (auto &p : this->particles) {
		particles.push_back(std::shared_ptr<MPMParticle>(p->duplicate()));
	}
    return particles;
}

real MPM::get_current_time() {
    return t;
}

LevelSet2D MPM::get_material_levelset() {
    return material_levelset;
}

void MPM::rasterize() {
    for (auto &p : particles) {
        if (p->state == MPMParticle::INACTIVE)
            continue;
        if (!is_normal(p->pos)) {
            p->print();
        }
        for (auto &ind : get_bounded_rasterization_region(p->pos)) {
            real weight = p->get_cache_w(ind);
            grid.mass[ind] += weight * p->mass;
            grid.velocity[ind] += weight * p->mass * (p->v + (3.0f) * p->b * (Vector2(ind.i, ind.j) - p->pos));
        }
    }
    grid.normalize_velocity();
}

void MPM::resample() {
    // FLIP is disabled here
    real alpha_delta_t = 1; // pow(flip_alpha, delta_t / flip_alpha_stride);
    if (apic)
        alpha_delta_t = 0.0f;
    for (auto &p : particles) {
        // Update particles with state UPDATING only
        if (p->state != MPMParticle::UPDATING)
            continue;
        real delta_t = base_delta_t * (t_int - p->last_update);
        Vector2 v = Vector2(0, 0), bv = Vector2(0, 0);
        Matrix2 cdg(0.0f);
        Matrix2 b(0.0f);
        int count = 0;
        for (auto &ind : get_bounded_rasterization_region(p->pos)) {
            count++;
            real weight = p->get_cache_w(ind);
            Vector2 gw = p->get_cache_gw(ind);
            Vector2 grid_vel = grid.velocity[ind] + grid.force_or_acc[ind] * delta_t;
            v += weight * grid_vel;
            Vector2 aa = grid_vel;
            Vector2 bb = Vector2(ind.i, ind.j) - p->pos;
            Matrix2 out(aa[0] * bb[0], aa[1] * bb[0], aa[0] * bb[1], aa[1] * bb[1]);
            b += weight * out;
            bv += weight * grid.velocity_backup[ind];
            cdg += glm::outerProduct(grid_vel, gw);
        }
        if (count != 16 || !apic) {
            b = Matrix2(0.0f);
        }
        CV(cdg);
        p->b = b;
        cdg = Matrix2(1.0f) + delta_t * cdg;

        p->v = (1 - alpha_delta_t) * v + alpha_delta_t * (v - bv + p->v);
        Matrix2 dg = cdg * p->dg_e * p->dg_p;
        p->dg_e = cdg * p->dg_e;
        p->dg_cache = dg;

        p->plasticity();
    }
}

void MPM::apply_deformation_force() {
    for (auto &p : particles) {
        if (p->state != MPMParticle::INACTIVE)
            p->calculate_force();
    }
    for (auto &p : particles) {
        if (p->state == MPMParticle::INACTIVE) {
            continue;
        }
        for (auto &ind : get_bounded_rasterization_region(p->pos)) {
            real mass = grid.mass[ind];
            if (mass == 0.0f) { // NO NEED for eps here
                continue;
            }
            Vector2 gw = p->get_cache_gw(ind);
            Vector2 force = p->tmp_force * gw;
            grid.force_or_acc[ind] += force;
        }
    }
}

TC_NAMESPACE_END

