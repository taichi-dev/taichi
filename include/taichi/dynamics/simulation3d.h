/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/common/meta.h>
#include <taichi/visualization/particle_visualization.h>
#include <vector>

TC_NAMESPACE_BEGIN

class Simulation3D : public Unit {
protected:
    real current_t = 0.0f;
    int num_threads;
public:
    Simulation3D() {}
    virtual real get_current_time() const {
        return current_t;
    }
    virtual void initialize(const Config &config) {
        num_threads = config.get_int("num_threads");
    }
    virtual void step(real t) {
        error("no impl");
    }
    virtual std::vector<RenderParticle> get_render_particles() const {
        error("no impl");
        return std::vector<RenderParticle>();
    }
    virtual void update(const Config &config) {}
};

TC_INTERFACE(Simulation3D);

TC_NAMESPACE_END
