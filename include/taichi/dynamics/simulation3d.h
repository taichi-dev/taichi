/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>
                  2017 Yu Fang <squarefk@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/meta.h>
#include <taichi/visualization/particle_visualization.h>
#include <vector>
#include <taichi/math/dynamic_levelset_3d.h>

TC_NAMESPACE_BEGIN

class Simulation3D : public Unit {
protected:
    real current_t = 0.0f;
    int num_threads;
    DynamicLevelSet3D levelset;
public:
    Simulation3D() {}

    virtual real get_current_time() const {
        return current_t;
    }

    virtual void initialize(const Config &config) override {
        num_threads = config.get_int("num_threads");
    }

    virtual void add_particles(const Config &config) {
        error("no impl");
    }

    virtual void step(real t) {
        error("no impl");
    }

    virtual std::vector<RenderParticle> get_render_particles() const {
        error("no impl");
        return std::vector<RenderParticle>();
    }

    virtual void set_levelset(const DynamicLevelSet3D &levelset) {
        this->levelset = levelset;
    }

    virtual void update(const Config &config) {}

    virtual bool test() const override {
        return true;
    };
};

TC_INTERFACE(Simulation3D);

TC_NAMESPACE_END
