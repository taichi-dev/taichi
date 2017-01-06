#pragma once

#include "euler_fluid.h"
#include "level_set.h"

TC_NAMESPACE_BEGIN

class FLIPFluid : public EulerFluid {
protected:
    Array u_backup;
    Array v_backup;
    Array u_count;
    Array v_count;
    float FLIP_alpha;
    float padding;
    int advection_order;
    float correction_strength;
    int correction_neighbours;

    void clamp_particle(Particle &p);

    virtual void initialize_solver(const Config &config);

    Vector2 sample_velocity(Vector2 position, Vector2 velocity, float lerp);

    virtual void advect(float delta_t);

    virtual void apply_external_forces(float delta_t);

    virtual void rasterize();

    template <float(*T)(const Particle &, const Vector2 &)>
    void rasterize_component(Array &val, Array &count);

    virtual void backup_velocity_field();

    virtual void substep(float delta_t);

    void reseed();

    void correct_particle_positions(float delta_t, bool clear_c = false);

public:

    FLIPFluid();

    void show(ImageBuffer<Vector3> &buffer);

    virtual void step(float delta_t);

};

TC_NAMESPACE_END

