#pragma once

#include <memory>
#include <vector>
#include "mpm_grid.h"
#include <taichi/math/levelset_2d.h>
#include <taichi/visualization/image_buffer.h>

TC_NAMESPACE_BEGIN

class MPM {
protected:
    Config config;
    std::vector<std::shared_ptr<Particle>> particles;
    Grid grid;
    int width;
    int height;
    real flip_alpha;
    real flip_alpha_stride;
    real h;
    real t;
    real last_sort;
    real sorting_period;
    vec2 gravity;
    bool apic;
    bool use_level_set;
    real max_delta_t;
    real min_delta_t;
    LevelSet2D levelset;
    LevelSet2D material_levelset;

    void compute_material_levelset();

    Region2D get_bounded_rasterization_region(Vector2 p) {
        int x = int(p.x);
        int y = int(p.y);
        int x_min = std::max(0, x - 1);
        int x_max = std::min(width, x + 3);
        int y_min = std::max(0, y - 1);
        int y_max = std::min(height, y + 3);
        return Region2D(x_min, x_max, y_min, y_max);
    }

    void particle_collision_resolution();

    void estimate_volume();

    void rasterize();

    void resample(real delta_t);

    void apply_deformation_force(real delta_t);

    virtual void substep(real delta_t);

    real get_dt_with_cfl_1();

    real get_max_speed();

    real cfl;

public:
    MPM() {
        sorting_period = 1.0f;
    }

    void initialize(const Config &config_);

    void step(real delta_t = 0.0f);

    void add_particle(const Config &config);

    void add_particle(std::shared_ptr<MPMParticle> particle);

    void add_particle(EPParticle p);

    void add_particle(DPParticle p);

    std::vector<std::shared_ptr<MPMParticle>> get_particles();

    real get_current_time();

    void set_levelset(const LevelSet2D &levelset) {
        this->levelset = levelset;
    }

    LevelSet2D get_material_levelset();
};

TC_NAMESPACE_END

