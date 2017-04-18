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
    scheduler.initialize(res, base_delta_t, cfl, &levelset);
    grid.initialize(res, &scheduler);
    particle_collision = config.get("particle_collision", true);
    position_noise = config.get("position_noise", 0.5f);
    if (async) {
        maximum_delta_t = config.get("maximum_delta_t", 1e-1f);
    } else {
        maximum_delta_t = base_delta_t;
    }
    material_levelset.initialize(res + Vector2i(1), Vector2(0));
    this->debug_input = config.get("debug_input", Vector4(0, 0, 0, 0));
    debug_blocks.initialize(scheduler.res, Vector4(0), Vector2(0));
}

void MPM::substep() {
    scheduler.update_particle_groups();
    scheduler.reset_particle_states();

    real delta_t = base_delta_t;

    grid.reset();
    int64 original_t_int_increment;
    int64 t_int_increment;

    if (async) {
        scheduler.reset();
        scheduler.update_dt_limits(t);

        original_t_int_increment = std::min(get_largest_pot(int64(maximum_delta_t / base_delta_t)),
                                   scheduler.update_max_dt_int(t_int));

        // t_int_increment is the biggest allowed dt.
        t_int_increment = original_t_int_increment - t_int % original_t_int_increment;

        if (debug_input[2] > 0) {
            P(t_int);
            P(t_int_increment);
        }
        t_int += t_int_increment; // final dt
        t = base_delta_t * t_int;

        scheduler.set_time(t_int);

        scheduler.expand(false, true);
        if (debug_input[2] > 0) {
            P(t_int_increment);
            scheduler.visualize(debug_input, debug_blocks);
            scheduler.print_max_dt_int();
        }
    } else {
        // Sync
        t_int_increment = 1;
        scheduler.states = 2;
        for (auto &p : particles) {
            p->state = MPMParticle::UPDATING;
            p->march_interval = 1;
        }
        t_int += t_int_increment; // final dt
        t = base_delta_t * t_int;
    }

    scheduler.update();

    for (auto &p : scheduler.get_active_particles()) {
        p->calculate_kernels();
    }

    rasterize();
    estimate_volume();
    grid.backup_velocity();
    apply_deformation_force();
    grid.apply_external_force(gravity);
    grid.normalize_acceleration();
    grid.apply_boundary_conditions(levelset, t_int_increment * base_delta_t, t);
    resample();
    for (auto &p : scheduler.get_active_particles()) {
        if (p->state == MPMParticle::UPDATING) {
            p->pos += (t_int - p->last_update) * delta_t * p->v;
            p->last_update = t_int;
            p->pos[0] = clamp(p->pos.x, 0.5f, res[0] - 0.5f);
            p->pos[1] = clamp(p->pos.y, 0.5f, res[1] - 0.5f);
        }
    }
    if (particle_collision)
        particle_collision_resolution();

    if (async) {
        scheduler.enforce_smoothness(original_t_int_increment);
    }
}

void MPM::kill_outside_particles() {
    // TODO: accelerate here
    std::vector<Particle *> new_particles;
    for (auto &p : particles) {
        bool killed = false;
        if (p->state == MPMParticle::UPDATING) {
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
        } else {
            delete p;
        }
    }
    particles.swap(new_particles);
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

void MPM::add_particle(std::shared_ptr<MPMParticle> p) {
    // WTH???
    p->mass = 1.0f / res[0] / res[0];
    p->pos += position_noise * Vector2(rand() - 0.5f, rand() - 0.5f);
    Particle *p_direct = p->duplicate();
    particles.push_back(p_direct);
    scheduler.insert_particle(p_direct);
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
    for (auto &p : scheduler.get_active_particles()) {
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
    for (auto &p : scheduler.get_active_particles()) {
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
    for (auto &p : scheduler.get_active_particles()) {
        p->calculate_force();
    }
    for (auto &p : scheduler.get_active_particles()) {
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

