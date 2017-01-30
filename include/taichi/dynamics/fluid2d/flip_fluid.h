#pragma once

#include "taichi/dynamics/fluid2d/euler_fluid.h"

TC_NAMESPACE_BEGIN

class FLIPFluid : public EulerFluid {
protected:
    Array u_backup;
    Array v_backup;
    Array u_count;
    Array v_count;
    real FLIP_alpha;
    real padding;
    int advection_order;
    real correction_strength;
    int correction_neighbours;

    void clamp_particle(Particle &p);

    virtual void initialize_solver(const Config &config);

    Vector2 sample_velocity(Vector2 position, Vector2 velocity, real lerp);

    virtual void advect(real delta_t);

    virtual void apply_external_forces(real delta_t);

    virtual void rasterize();

    template <real(*T)(const Particle &, const Vector2 &)>
    void rasterize_component(Array &val, Array &count);

    virtual void backup_velocity_field();

    virtual void substep(real delta_t);

    void reseed();

    void correct_particle_positions(real delta_t, bool clear_c = false);

public:

    FLIPFluid();

    void show(Array2D<Vector3> &buffer);

    virtual void step(real delta_t);

};

TC_NAMESPACE_END

