#include "euler_smoke.h"

TC_NAMESPACE_BEGIN

void EulerSmoke::apply_external_forces(real delta_t)
{
    for (auto &ind : v.get_region()) {
        if (boundary_levelset.sample(ind) > 0) {
            real force = -buoyancy_alpha * density[ind] + buoyancy_beta * temperature[ind];
            v[ind] += delta_t * force;
        }
    }
}

void EulerSmoke::initialize(const Config & config)
{
    EulerFluid::initialize(config);
    buoyancy_alpha = config.get_real("buoyancy_alpha");
    buoyancy_beta = config.get_real("buoyancy_beta");
    density = Array(width, height, 0);
    temperature = Array(width, height, 0);
}

void EulerSmoke::emit(real delta_t)
{
    for (auto &emit : sources) {
        auto c = emit.get_vec2("center");
        auto r = emit.get_real("radius");
        auto init_temperature = emit.get_real("temperature");
        auto init_density = emit.get_real("density");
        auto init_v = emit.get_vec2("velocity");
        auto failure_limit = emit.get("failure_limit", 100);
        auto emission = emit.get_real("emission") * delta_t;
        int num_particles = int(emission) + (rand() < (delta_t - floor(delta_t)));
        for (int i = 0; i < num_particles; i++) {
            int failures = 0;
            while (failures < failure_limit) {
                real d = sqrt(rand()) * r, phi = rand() * 2 * pi;
                Vector2 position = c + d * Vector2(sin(phi), cos(phi));
                if (boundary_levelset.sample(position) < 0) {
                    failures++;
                    continue;
                }
                particles.push_back(Particle(position, init_v));
                break;
            }
            if (failures == failure_limit) {
                printf("Warning: too many emission failures. (Make sure the source region is within the boundary)\n");
            }
        }
        for (auto &ind : temperature.get_region()) {
            if (boundary_levelset.sample(ind.get_pos()) > 0) {
                if (length(ind.get_pos() - c) < r) {
                    temperature[ind] = init_temperature;
                    density[ind] = init_density;
                }
            }
        }

    }
}

void EulerSmoke::substep(real delta_t)
{
    emit(delta_t);
    apply_external_forces(delta_t);
    cell_types = CellType::WATER;
    project(delta_t);
    advect(delta_t);
    t += delta_t;

    density = advect(density, delta_t);
    temperature = advect(temperature, delta_t);
}

TC_IMPLEMENTATION(Fluid, EulerSmoke, "smoke");

TC_NAMESPACE_END
