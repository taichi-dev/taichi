/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>
#include "mpm_scheduler.h"
#include "mpm_grid.h"
#include <taichi/math/levelset_2d.h>
#include <taichi/math/dynamic_levelset_2d.h>
#include <taichi/visual/texture.h>
#include <taichi/visualization/image_buffer.h>

TC_NAMESPACE_BEGIN

extern long long kernel_calc_counter;

class MPM : public Unit {
protected:
    Vector2i res;
    Grid grid;
    std::vector<Particle *> particles;

    real flip_alpha;
    real flip_alpha_stride;
    real h;
    real t;
    real base_delta_t;
    real maximum_delta_t;
    real requested_t;
    int64 t_int;
    real cfl;
    real strength_dt_mul;

    Vector2 gravity;
    Vector4 debug_input;

    MPMScheduler scheduler;

    DynamicLevelSet2D levelset;
    LevelSet2D material_levelset;

    real position_noise;
    bool particle_collision;
    bool async;
    bool apic;
    bool kill_at_boundary;
    Array2D<Vector4> debug_blocks;

    void compute_material_levelset();

    Region2D get_bounded_rasterization_region(Vector2 p) {
        int x = int(p.x);
        int y = int(p.y);
        int x_min = std::max(0, x - 1);
        int x_max = std::min(res[0], x + 3);
        int y_min = std::max(0, y - 1);
        int y_max = std::min(res[1], y + 3);
        return Region2D(x_min, x_max, y_min, y_max);
    }

    void particle_collision_resolution();

    void estimate_volume();

    void rasterize();

    void apply_deformation_force(real delta_t);

    void resample(real grid_delta_t);

    virtual void substep();

public:
    MPM() {}

    void initialize(const Config &config) override;

    void step(real delta_t = 0.0f);

    void add_particle(const Config &config);

    void add_particle(std::shared_ptr<MPMParticle> particle);

    void add_particle(EPParticle p);

    void add_particle(DPParticle p);

    std::vector<std::shared_ptr<Particle>> get_particles();

    real get_current_time();

    void set_levelset(const DynamicLevelSet2D &levelset) {
        this->levelset = levelset;
    }

    LevelSet2D get_material_levelset();

    Array2D<Vector4> get_debug_blocks() {
        return debug_blocks;
    }

    int get_grid_block_size() {
        return grid_block_size;
    }

    void kill_outside_particles();

    bool test() const override;
};

TC_NAMESPACE_END

