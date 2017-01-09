#pragma once

#include "fluid.h"
#include <taichi/system/timer.h>
#include <taichi/math/array_2d.h>
#include "point_level_set.h"
#include <memory>
#include <taichi/visualization/image_buffer.h>
#include <taichi/math/stencils.h>
#include <taichi/levelset/levelset2d.h>

TC_NAMESPACE_BEGIN

class EulerFluid : public Fluid {
protected:
    int kernel_size;
    Array Ad, Ax, Ay, E;
    Array2D<int> water_cell_index;
    void apply_pressure(const Array &p);
    Array apply_A(const Array &x);
    Array apply_preconditioner(const Array &x);
    Array get_rhs();
    void apply_boundary_condition();
    float volume_correction_factor;
    bool supersampling;
    bool show_grid;
    float cfl;
    std::string title;
    Array u_weight;
    Array v_weight;
    LevelSet2D liquid_levelset;

    int width, height;
    Array pressure, q, z;
    float target_water_cells;
    float last_water_cells;
    float integrate_water_cells_difference;
    void initialize_volume_controller();
    void update_volume_controller();
    float get_volume_correction();
    void advect_level_set(float delta_t);
    bool use_bridson_pcg;
    float tolerance;
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
    float t;
    enum CellType {
        AIR = 0, WATER = 1
    };
    Array2D<CellType> cell_types;

    Vector2 sample_velocity(Vector2 position, const Array &u, const Array &v);

    virtual Vector2 sample_velocity(Vector2 position);

    std::function<CellType(float, float)> get_initializer(std::string name);

    bool check_u_activity(int i, int j);

    bool check_v_activity(int i, int j);

    virtual void simple_extrapolate();

    void level_set_extrapolate();
    
    bool inside(int x, int y);

    virtual void prepare_for_pressure_solve();

    virtual Array solve_pressure_naive();

    virtual void project(float delta_t);

    virtual void apply_viscosity(float delta_t);

    int count_water_cells();

    virtual void mark_cells();

    virtual void show_surface();

    Vector2 sl_position(Vector2 position, float delta_t);

    void print_u();

    void print_v();
     
    void initialize_pressure_solver();

    Vector2 clamp_particle_position(Vector2 pos);
    
    void advect(float delta_t);

    virtual void apply_external_forces(float delta_t);

    static Vector2 position_noise();

    static float kernel(const Vector2 &c) {
        return max(0.0f, 1.0f - std::abs(c.x)) * max(0.0f, 1.0f - std::abs(c.y));
    }

    static Vector2 grad_kernel(const Vector2 &c) {
#define PRECISE_SGN(x) ((-1 < x && x <= 0) ? -1 : ((0 < x && x <= 1) ? 1 : 0))
        return Vector2(
            PRECISE_SGN(c.x) * max(0.0f, 1.0f - abs(c.y)), 
            PRECISE_SGN(c.y) * max(0.0f, 1.0f - abs(c.x))
            );
#undef PRECISE_SGN
    }

    virtual void initialize_solver(const Config &config);

    virtual void initialize_particles(const Config &config);

    virtual void substep(float delta_t);
    
    virtual float get_dt_with_cfl_1();

    virtual float get_max_grid_speed();

    virtual void compute_liquid_levelset();

    virtual Array advect(const Array &arr, float delta_t);

    virtual bool check_diag_domination();

public:
    
    EulerFluid();

    virtual void set_levelset(const LevelSet2D &boundary_levelset);

    virtual void initialize(const Config &config);

    virtual void step(float delta_t);
    
    virtual void show(Array2D<Vector3> &buffer);

    virtual float get_current_time();

    virtual void add_particle(Particle &particle);

    virtual std::vector<Fluid::Particle> get_particles();

    virtual LevelSet2D get_liquid_levelset();

    Array get_density();

    virtual void add_source(const Config &config);

    virtual Array get_pressure();
};


TC_NAMESPACE_END

