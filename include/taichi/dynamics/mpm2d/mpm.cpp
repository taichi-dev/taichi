#include "mpm.h"

TC_NAMESPACE_BEGIN

void MPM::initialize(const Config &config_) {
    auto config = Config(config_);
    this->config = config;
    width = config.get_int("simulation_width");
    height = config.get_int("simulation_height");
    this->apic = config.get("apic", true);
    this->use_level_set = config.get("use_level_set", false);
    this->cfl = config.get("cfl", 0.01f);
    this->h = config.get_real("delta_x");
    grid.initialize(width, height);
    t = 0.0f;
    last_sort = 1e20f;
    flip_alpha = config.get_real("flip_alpha");
    flip_alpha_stride = config.get_real("flip_alpha_stride");
    gravity = config.get_vec2("gravity");
    max_delta_t = config.get("max_delta_t", 0.001f);
    min_delta_t = config.get("min_delta_t", 0.00001f);
    material_levelset.initialize(width, height, Vector2(0.5f, 0.5f));
}

void MPM::substep(real delta_t) {
    if (!particles.empty()) {
        for (auto &p : particles) {
            p->calculate_kernels();
        }
        rasterize();
        grid.reorder_grids();
        estimate_volume();
        grid.backup_velocity();
        grid.apply_external_force(gravity, delta_t);
        //Deleted: grid.apply_boundary_conditions(levelset);
        apply_deformation_force(delta_t);
        grid.apply_boundary_conditions(levelset);
        resample(delta_t);
        for (auto &p : particles) {
            p->pos += delta_t * p->v;
        }
        if (config.get("particle_collision", false))
            particle_collision_resolution();
    }
    t += delta_t;
}

void MPM::step(real delta_t) {
    real simulation_time = 0.0f;
    while (simulation_time < delta_t - eps) {
        real purpose_dt = std::min(max_delta_t, get_dt_with_cfl_1() * cfl);
        real thres = min_delta_t;
        if (purpose_dt < delta_t * thres) {
            purpose_dt = delta_t * thres;
            printf("substep dt too small, clamp.\n");
        }
        real dt = std::min(delta_t - simulation_time, purpose_dt);
        substep(dt);
        simulation_time += dt;
    }
    compute_material_levelset();
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
        p->resolve_collision(levelset);
    }
}

void MPM::estimate_volume() {
    for (auto &p : particles) {
        if (p->vol == -1.0f) {
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
    auto p = create_particle(config);
    p->mass = 1.0f / width / width;
    // p->pos += config.get("position_noise", 0.0f) * Vector2(rand() - 0.5f, rand() - 0.5f);
    particles.push_back(p);
}

void MPM::add_particle(std::shared_ptr<MPMParticle> p) {
    p->mass = 1.0f / width / width;
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

inline void MPM::resample(real delta_t) {
    real alpha_delta_t = pow(flip_alpha, delta_t / flip_alpha_stride);
    if (apic)
        alpha_delta_t = 0.0f;
    for (auto &p : particles) {
        int p_i = int(p->pos.x);
        int p_j = int(p->pos.y);
        Vector2 v = Vector2(0, 0), bv = Vector2(0, 0);
        mat2 cdg(0.0f);
        mat2 b(0.0f);
        int count = 0;
        for (auto &ind : get_bounded_rasterization_region(p->pos)) {
            count++;
            real weight = p->get_cache_w(ind);
            Vector2 gw = p->get_cache_gw(ind);
            v += weight * grid.velocity[ind];
            Vector2 aa = grid.velocity[ind];
            Vector2 bb = Vector2(ind.i, ind.j) - p->pos;
            mat2 out(aa[0] * bb[0], aa[1] * bb[0], aa[0] * bb[1], aa[1] * bb[1]);
            b += weight * out;
            bv += weight * grid.velocity_backup[ind];
            cdg += glm::outerProduct(grid.velocity[ind], gw);
        }
        if (count != 16 || !apic) {
            b = mat2(0.0f);
        }
        CV(cdg);
        p->b = b;
        cdg = mat2(1.0f) + delta_t * cdg;

        p->v = (1 - alpha_delta_t) * v + alpha_delta_t * (v - bv + p->v);
        mat2 dg = cdg * p->dg_e * p->dg_p;
        p->dg_e = cdg * p->dg_e;
        p->dg_cache = dg;
    }
    for (auto &p : particles) {
        p->plasticity();
    }
}

void MPM::apply_deformation_force(real delta_t) {
#pragma omp parallel for
    for (auto &p : particles) {
        p->calculate_force();
    }
    // NOTE: Potential racing condition errors!
#pragma omp parallel for
    for (auto &p : particles) {
        for (auto &ind : get_bounded_rasterization_region(p->pos)) {
            real mass = grid.mass[ind];
            if (mass == 0.0f) { // NO NEED for eps here
                continue;
            }
            Vector2 gw = p->get_cache_gw(ind);
            Vector2 force = p->tmp_force * gw;
            grid.velocity[ind] += delta_t / mass * force;
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

