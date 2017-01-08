#pragma once

#include <memory.h>
#include <string>
#include <taichi/visualization/image_buffer.h>
#include <taichi/common/meta.h>
#include <taichi/common/interface.h>
#include <taichi/math/array_3d.h>
#include <taichi/dynamics/poisson_solver3d.h>
#include <taichi/dynamics/simulation3d.h>

TC_NAMESPACE_BEGIN

class Tracker3D {
public:
    Vector3 position;
    Vector3 color;
    Tracker3D() {}
    Tracker3D(const Vector3 &position, const Vector3 &color) : position(position), color(color) {}
};

class Smoke3D : public Simulation3D {
    typedef Array3D<real> Array;
public:
    Array u, v, w, rho, t, pressure, last_pressure;
    Vector3i res;
    real smoke_alpha, smoke_beta;
    real temperature_decay;
    real pressure_tolerance;
    real density_scaling;
    Vector3 initial_speed;
    real tracker_generation;
    real perturbation;
    bool open_boundary;
    std::vector<Tracker3D> trackers;
    std::shared_ptr<PoissonSolver3D> pressure_solver;
    PoissonSolver3D::BCArray boundary_condition;

    Smoke3D() {}

    void remove_outside_trackers();

    void initialize(const Config &config);

    void project();

    void confine_vorticity(real delta_t);

    void advect(real delta_t);

    void move_trackers(real delta_t);

    void step(real delta_t);

    virtual void show(ImageBuffer<Vector3> &buffer);

    void advect(Array &attr, real delta_t);

    void apply_boundary_condition();

    static Vector3 sample_velocity(const Array &u, const Array &v, const Array &w, const Vector3 &pos);

    Vector3 sample_velocity(const Vector3 &pos) const;

    std::vector<RenderParticle> get_render_particles() const;
};

TC_NAMESPACE_END
