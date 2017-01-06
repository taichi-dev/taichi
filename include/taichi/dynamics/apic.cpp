#include "apic.h"

TC_NAMESPACE_BEGIN

APICFluid::APICFluid()
{
}

void APICFluid::initialize_solver(const Config &config)
{
    FLIPFluid::initialize_solver(config);
    FLIP_alpha = 0.0f;
    padding = config.get("padding", 0.501f);
    advection_order = config.get("advection_order", 1);
    if (advection_order == 2) {
        printf("Warning: using second order advection can be unstable for APIC!\n");
    }
    apic_blend = config.get("apic_blend", 1.0f);
    printf("initialized\n");
}

void APICFluid::rasterize()
{
    rasterize_component<Particle::get_affine_velocity<0>>(u, u_count);
    rasterize_component<Particle::get_affine_velocity<1>>(v, v_count);
}

void APICFluid::sample_c()
{
    for (auto &p : particles) {
        p.c[0] = apic_blend * sample_c(p.position, u);
        p.c[1] = apic_blend * sample_c(p.position, v);
    }
}

Vector2 APICFluid::sample_c(Vector2 & pos, Array & val) {
    const int extent = (1 + 1) / 2;
    Vector2 c(0);
    for (auto &ind : val.get_rasterization_region(pos, extent)) {
        if (!val.inside(ind)) continue;
        Vector2 grad = grad_kernel(ind.get_pos() - pos);
        c += grad * val[ind];
    }
    return c;
}

void APICFluid::substep(float delta_t)
{
    Time::Timer _("substep");
    apply_external_forces(delta_t);
    mark_cells();
    rasterize();
    compute_liquid_levelset();
    simple_extrapolate();
    TIME(project(delta_t));
    simple_extrapolate();
    apply_boundary_condition();
    sample_c();
    advect(delta_t);
    t += delta_t;
}

TC_NAMESPACE_END

