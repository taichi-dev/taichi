/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once
#include <taichi/dynamics/fluid2d/flip_liquid.h>
#include <taichi/nearest_neighbour/point_cloud.h>

TC_NAMESPACE_BEGIN

class FLIPSmoke : public FLIPLiquid {
protected:
    real ambient_temp;
    real buoyancy;
    real conduction;
    Array temperature;
    Array temperature_backup;
    Array temperature_count;
    real source_temperature;
    real temperature_flip_alpha;
    std::string visualization;
    virtual void initialize(const Config &config) override {
        FLIPLiquid::initialize(Config(config).set("initializer", "full"));

        temperature = Array(width, height);
        temperature_backup = Array(width, height);
        temperature_count = Array(width, height);

        buoyancy = config.get("bouyancy", 0.1f);
        conduction = config.get("conduction", 0.0f);
        visualization = config.get("visualization", "");
        source_temperature = config.get("source_temperature", ambient_temp);
        temperature_flip_alpha = config.get("temperature_flip_alpha", 0.97f);
        gravity = Vector2(0, 0.5f * height);
        ambient_temp = 200;

        for (auto &p : particles) {
            p.temperature = ambient_temp;
        }
    }
    virtual void simple_mark_cells() {
        cell_types = CellType::WATER;
    }
    void seed_particles(real delta_t) {
        for (int i = 0; i < 100; i++) {
            Vector2 pos((0.4f + 0.2f * rand()) * width, (0.1f + 0.1f * rand()) * height);
            Vector2 vel(0.0f, 0.0f);
            Particle p(pos, vel);
            p.temperature = source_temperature;
            particles.push_back(p);
        }
    }
    void apply_external_forces(real delta_t) {
        for (auto &p : particles) {
            p.velocity += buoyancy * delta_t * Vector2(0, 1) * (p.temperature - ambient_temp);
        }
    }
    virtual void step(real delta_t) {
        seed_particles(delta_t);
        FLIPLiquid::step(delta_t);
    }
    virtual void substep(real delta_t) {
        apply_external_forces(delta_t);
        simple_mark_cells();
        rasterize();
        rasterize_temperature();
        extrapolate();
        temperature_backup = temperature;
        backup_velocity_field();
        apply_boundary_condition();
        project(delta_t);
        advect(delta_t);
        if (conduction > 0)
            diffuse_temperature(delta_t);
        resample_temperature(delta_t);
        t += delta_t;
    }
    virtual void voronoi_extrapolate(Array &val, const Array &weight) {
        NearestNeighbour2D voronoi;
        std::vector<Vector2> points;
        std::vector<real> values;
        for (auto ind : val.get_region()) {
            if (weight[ind] > 0) {
                points.push_back(Vector2(real(ind.i), real(ind.j)));
                values.push_back(val[ind]);
            }
        }
        voronoi.initialize(points);
        for (auto ind : val.get_region()) {
            if (weight[ind] == 0) {
                val[ind] = values[voronoi.query_index(Vector2(real(ind.i), real(ind.j)))];
            }
        }
    }
    virtual void extrapolate() {
        voronoi_extrapolate(u, u_count);
        voronoi_extrapolate(v, v_count);
        voronoi_extrapolate(temperature, temperature_count);
    }
    virtual void voronoi_rasterize() {
        NearestNeighbour2D voronoi[2];
        std::vector<Vector2> points[2];
        std::vector<real> values[2];
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < (int)particles.size(); i++) {
                points[k].push_back(particles[i].position);
                values[k].push_back(particles[i].velocity[k]);
            }
            voronoi[k].initialize(points[k]);
        }
        for (int i = 0; i < width + 1; i++) {
            for (int j = 0; j < height; j++) {
                u[i][j] = values[0][voronoi[0].query_index(Vector2((real)i, j + 0.5f))];
            }
        }
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height + 1; j++) {
                v[i][j] = values[1][voronoi[1].query_index(Vector2(i + 0.5f, (real)j))];
            }
        }
    }
    void rasterize_temperature() {
        temperature = 0;
        temperature_count = 0;
        for (auto &p : particles) {
            int x, y;
            x = (int)floor(p.position.x - 0.5);
            y = (int)floor(p.position.y - 0.5);
            for (int dx = 0; dx < 2; dx++) {
                for (int dy = 0; dy < 2; dy++) {
                    int nx = x + dx, ny = y + dy;
                    if (!temperature.inside(nx, ny)) {
                        continue;
                    }
                    real weight = kernel(p.position - vec2(nx + 0.5f, ny + 0.5f));
                    temperature[nx][ny] += weight * p.temperature;
                    temperature_count[nx][ny] += weight;
                }
            }
        }
        for (auto ind : temperature.get_region()) {
            if (temperature_count[ind] > 0) {
                temperature[ind] /= temperature_count[ind];
            }
            else {
                // extrapolation...
            }
        }
    }
    void diffuse_temperature(real delta_t) {
        real exchange = conduction * delta_t;
        Array new_temperature = temperature;
        for (auto ind : temperature.get_region()) {
            for (auto d : neighbour4) {
                auto nei = ind.neighbour(d);
                if (temperature.inside(nei)) {
                    real delta = exchange * temperature[nei];
                    new_temperature[ind] += delta;
                    new_temperature[nei] -= delta;
                }
            }
        }
        temperature = new_temperature;
    }

    void resample_temperature(real delta_t) {
        real alpha = powf(temperature_flip_alpha, delta_t / 0.01f);
        for (auto &p : particles) {
            p.temperature = alpha * (-temperature_backup.sample(p.position.x, p.position.y) + p.temperature) + temperature.sample(p.position.x, p.position.y);
        }
    }
};

TC_NAMESPACE_END
