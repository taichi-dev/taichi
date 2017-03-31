/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm.h"
#include <taichi/common/asset_manager.h>

TC_NAMESPACE_BEGIN

long long kernel_calc_counter = 0;

void MPM::initialize(const Config &config_) {
    auto config = Config(config_);
    this->config = config;
    res = config.get_vec2i("res");
    this->apic = config.get("apic", true);
    this->use_level_set = config.get("use_level_set", false);
    this->cfl = config.get("cfl", 0.01f);
    this->h = config.get_real("delta_x");
    int dt_multiplier_id = config.get("dt_multiplier_tex_id", -1);
    if (dt_multiplier_id != -1) {
        this->dt_multiplier = AssetManager::get_asset<Texture>(dt_multiplier_id);
    } else {
        Config cfg;
        cfg.set("value", Vector4(1.0f));
        this->dt_multiplier = create_instance<Texture>("const", cfg);
    }
    grid.initialize(res);
    t = 0.0f;
    t_int = 0;
    requested_t = 0.0f;
    last_sort = 1e20f;
    flip_alpha = config.get_real("flip_alpha");
    flip_alpha_stride = config.get_real("flip_alpha_stride");
    gravity = config.get_vec2("gravity");
    base_delta_t = config.get_real("base_delta_t");
    material_levelset.initialize(res, Vector2(0.5f, 0.5f));
}

void MPM::substep() {
    t_int += 1;
    t = base_delta_t * t_int;
    real delta_t = base_delta_t;

    bool exist_active_particle = false;
    Vector2 downscale(1.0f / res[0], 1.0f / res[1]);
    for (auto &p : particles) {
        Vector3 coord(p->pos.x * downscale.x, p->pos.y * downscale.y, 0.0f);
        p->march_interval = int(std::round(this->dt_multiplier->sample(coord).x));
        p->active = t_int % p->march_interval == 0;
        if (p->active) {
            p->calculate_kernels();
            exist_active_particle = true;
        }
    }
    if (!exist_active_particle) {
        return;
    }
    rasterize();
    grid.reorder_grids();
    estimate_volume();
    grid.backup_velocity();
    grid.apply_external_force(gravity);
    //Deleted: grid.apply_boundary_conditions(levelset);
    apply_deformation_force();
    grid.apply_boundary_conditions(levelset);
    grid.normalize_acceleration();
    resample();
    for (auto &p : particles) {
        if (p->active) {
            p->pos += p->march_interval * delta_t * p->v;
        }
    }
    if (config.get("particle_collision", false))
        particle_collision_resolution();
}

void MPM::step(real delta_t) {
    requested_t += delta_t;
    while (t + base_delta_t < requested_t)
        substep();
    compute_material_levelset();
    P(kernel_calc_counter);
}

void MPM::compute_material_levelset()
{
    material_levelset.reset(std::numeric_limits<real>::infinity());
    for (auto &p : particles) {
        for (auto &ind : material_levelset.get_rasterization_region(p->pos, 3)) {
            Vector2 delta_pos = ind.get_pos() - p->pos;
            material_levelset[ind] = std::min(material_levelset[ind], length(delta_pos) - 0.8f);
        }
    }
    for (auto &ind : material_levelset.get_region()) {
        if (material_levelset[ind] < 0.5f) {
            if (levelset.sample(ind.get_pos()) < 0)
                material_levelset[ind] = -0.5f;
        }
    }
}

void MPM::particle_collision_resolution() {
    for (auto &p : particles) {
        if (p->active)
            p->resolve_collision(levelset);
    }
}

void MPM::estimate_volume() {
    for (auto &p : particles) {
        if (p->active && p->vol == -1.0f) {
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
    // WTH???
    auto p = create_particle(config);
    p->mass = 1.0f / res[0] / res[0];
    // p->pos += config.get("position_noise", 0.0f) * Vector2(rand() - 0.5f, rand() - 0.5f);
    particles.push_back(p);
}

void MPM::add_particle(std::shared_ptr<MPMParticle> p) {
    // WTH???
    p->mass = 1.0f / res[0] / res[0];
    p->pos += config.get("position_noise", 0.0f) * Vector2(rand() - 0.5f, rand() - 0.5f);
    particles.push_back(p);
}

void MPM::add_particle(EPParticle p) {
    add_particle(std::make_shared<EPParticle>(p));
}

void MPM::add_particle(DPParticle p) {
    add_particle(std::make_shared<DPParticle>(p));
}

std::vector<std::shared_ptr<MPMParticle>> MPM::get_particles() {
    return particles;
}

real MPM::get_current_time() {
    return t;
}

LevelSet2D MPM::get_material_levelset() {
    return material_levelset;
}

void MPM::rasterize() {
    grid.reset();
    for (auto &p : particles) {
        if (!p->active)
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
        if (!p->active)
            continue;
        real delta_t = base_delta_t * p->march_interval;
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
        if (p->active)
            p->calculate_force();
    }
    for (auto &p : particles) {
        if (!p->active) {
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

real MPM::get_dt_with_cfl_1() {
    return 1 / max(get_max_speed(), 1e-5f);
}

real MPM::get_max_speed() {
    real maximum_speed = 0;
    for (auto &p : particles) {
        maximum_speed = max(abs(p->v.x), maximum_speed);
        maximum_speed = max(abs(p->v.y), maximum_speed);
    }
    return maximum_speed;
}

TC_NAMESPACE_END

