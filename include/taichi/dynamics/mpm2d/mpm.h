#pragma once

#include <memory>
#include <vector>
#include "mpm_grid.h"
#include <taichi/math/levelset_2d.h>
#include <taichi/visualization/image_buffer.h>

TC_NAMESPACE_BEGIN

#define CACHE_INDEX ((i - p_i + 1) * 4 + (j - p_j + 1))

struct MPMLinearSystemRow {
    int num_items;
    Matrix2 items[49];
    int indices[49];
    void reset() {
        num_items = 0;
    }
    void append(int index, const Matrix2 &item) {
        /*
        int i = 0;
        if (num_items > 0)
            for (i = num_items; i >= 1 && indices[i - 1] > index; i--) {
                indices[i] = indices[i - 1];
                items[i] = items[i - 1];
            }
        indices[i] = index;
        items[i] = item;
        num_items++;
        */
        indices[num_items] = index;
        items[num_items] = item;
        num_items++;
    }
};

class MPMLinearSystem {
public:
    std::vector<MPMLinearSystemRow> data;
    ArrayVec2 rhs;
    std::vector<Matrix2> diag;
    int size;
    void reset(int size) {
        this->size = size;
        data.resize(size);
        diag.resize(size);
        for (int i = 0; i < size; i++) {
            data[i].reset();
        }
    }
    ArrayVec2 apply(const ArrayVec2 &x) {
        // Time::TickTimer _("apply system");
        ArrayVec2 y(size, vec2(0));
        for (int i = 0; i < size; i++) {
            vec2 tmp_0(0), tmp_1(0);
            const int &num_items = data[i].num_items;
            for (int j = 0; j < num_items; j += 1) {
                tmp_0 += data[i].items[j] * x[data[i].indices[j]];
            }
            y[i] = tmp_0 + tmp_1;
            CV(y[i]);
        }
        return y;
    }
    void append(int row, int column, const Matrix2 &item) {
        data[row].append(column, item);
        if (row == column) {
            diag[row] = item;
        }
    }
    void precondition() {
        for (int i = 0; i < size; i++) {
            Matrix2 precond = glm::inverse(diag[i]);
            const int &num_items = data[i].num_items;
            for (int j = 0; j < num_items; j += 1) {
                data[i].items[j] *= precond;
            }
            rhs[i] = precond * rhs[i];
        }
    }

};

class MPM {
protected:
    Config config;
    std::vector<std::shared_ptr<Particle>> particles;
    Grid grid;
    int dim;
    int width;
    int height;
    real flip_alpha;
    real flip_alpha_stride;
    real h;
    real t;
    real last_sort;
    real sorting_period;
    vec2 gravity;
    real implicit_ratio;
    MPMLinearSystem system;
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

    mat4 get_energy_second_derivative_brute_force(Particle &p, real delta = 1e-2f);

    mat4 get_energy_second_derivative(Particle &p);

    void build_system(const real delta_t);

    void apply_A(const ArrayVec2 &x, ArrayVec2 &p);

    // CR solver
    ArrayVec2 solve_system(ArrayVec2 x_0, Grid &grid);

    void implicit_velocity_update(const real &delta_t);

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

