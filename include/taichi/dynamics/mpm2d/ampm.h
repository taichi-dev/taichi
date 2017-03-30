/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include "mpm.h"

TC_NAMESPACE_BEGIN

class AMPM : public MPM {
protected:
    Config config;
    Grid grid;

    std::vector<std::shared_ptr<Particle>> particles;

public:
    AMPM() : MPM() {
    }

    void initialize(const Config &config_) {
        MPM::initialize(config);

    }

    void substep(real delta_t = 0.0f) override {
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

};

TC_NAMESPACE_END

