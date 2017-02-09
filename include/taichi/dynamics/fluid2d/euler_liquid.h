#pragma once

#include "fluid.h"
#include <taichi/system/timer.h>
#include <taichi/math/array_2d.h>
#include <memory>
#include <taichi/visualization/image_buffer.h>
#include <taichi/math/stencils.h>
#include <taichi/math/levelset_2d.h>

TC_NAMESPACE_BEGIN

class EulerLiquid : public Fluid {
protected:
    int kernel_size;
    Array Ad, Ax, Ay, E;
    Array2D<int> water_cell_index;
    void apply_pressure(const Array &p);
    Array apply_A(const Array &x);
    Array apply_preconditioner(const Array &x);
    Array get_rhs();
    void apply_boundary_condition();
    real volume_correction_factor;
    real levelset_band;
    bool supersampling;
    real cfl;
    Array u_weight;
    Array v_weight;
    LevelSet2D liquid_levelset;

    int width, height;
    Array pressure, q, z;
    real target_water_cells;
    real last_water_cells;
    real integrate_water_cells_difference;
    void initialize_volume_controller();
    void update_volume_controller();
    real get_volume_correction();
    void advect_level_set(real delta_t);
    real tolerance;
    int maximum_iterations;
    LevelSet2D boundary_levelset;
    Array density;
    std::vector<Config> sources;

    const Vector2 supersample_positions[9]
    { Vector2(0.25f, 0.25f), Vector2(0.25f, 0.75f), Vector2(0.75f, 0.25f),
     Vector2(0.75f, 0.75f), Vector2(0.25f, 0.5f), Vector2(0.5f, 0.5f),
     Vector2(0.5f, 0.25f), Vector2(0.5f, 0.75f), Vector2(0.75f, 0.5f) };

    Array u, v, p;
    Vector2 gravity;
    real t;
    enum CellType {
        AIR = 0, WATER = 1
    };
    Array2D<CellType> cell_types;

    Vector2 sample_velocity(Vector2 position, const Array &u, const Array &v);

    virtual Vector2 sample_velocity(Vector2 position);

    std::function<CellType(real, real)> get_initializer(std::string name);

    bool check_u_activity(int i, int j);

    bool check_v_activity(int i, int j);

    virtual void simple_extrapolate();

    void level_set_extrapolate();
    
    bool inside(int x, int y);

    virtual void prepare_for_pressure_solve();

    virtual Array solve_pressure_naive();

    virtual void project(real delta_t);

    virtual void apply_viscosity(real delta_t);

    int count_water_cells();

    virtual void mark_cells();

    Vector2 sl_position(Vector2 position, real delta_t);

    void print_u();

    void print_v();
     
    void initialize_pressure_solver();

    Vector2 clamp_particle_position(Vector2 pos);
    
    void advect(real delta_t);

    virtual void apply_external_forces(real delta_t);

    static Vector2 position_noise();

    static real kernel(const Vector2 &c) {
        return max(0.0f, 1.0f - std::abs(c.x)) * max(0.0f, 1.0f - std::abs(c.y));
    }

    virtual void advect_liquid_levelset(real delta_t);

    virtual void rebuild_levelset(LevelSet2D &levelset, real band);

    static Vector2 grad_kernel(const Vector2 &c) {
#define PRECISE_SGN(x) ((-1 < x && x <= 0) ? -1 : ((0 < x && x <= 1) ? 1 : 0))
        return Vector2(
            PRECISE_SGN(c.x) * max(0.0f, 1.0f - abs(c.y)), 
            PRECISE_SGN(c.y) * max(0.0f, 1.0f - abs(c.x))
            );
#undef PRECISE_SGN
    }

    virtual void initialize_solver(const Config &config);

    virtual void substep(real delta_t);
    
    virtual real get_dt_with_cfl_1();

    virtual real get_max_grid_speed();

    virtual void compute_liquid_levelset();

    virtual Array advect(const Array &arr, real delta_t);

    virtual bool check_diag_domination();

public:
    
    EulerLiquid() {}

    virtual void set_levelset(const LevelSet2D &boundary_levelset) override;

    virtual void initialize(const Config &config) override;

    virtual void step(real delta_t) override;
    
    virtual real get_current_time() override;

    virtual void add_particle(Particle &particle) override;

    virtual std::vector<Fluid::Particle> get_particles() override;

    virtual LevelSet2D get_liquid_levelset() override;

    virtual Array get_density() override;

    virtual void add_source(const Config &config) override;

    virtual Array get_pressure() override;
};


TC_NAMESPACE_END

