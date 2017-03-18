#pragma once

#include "mpm_utils.h"
#include <stb_image.h>
#include <algorithm>
#include <atomic>
#include <taichi/math/array_2d.h>
#include <taichi/math/array_1d.h>
#include <taichi/math/levelset_2d.h>
#include "mpm_particle.h"

TC_NAMESPACE_BEGIN

typedef MPMParticle Particle;

extern Vector2 particle_offsets[4];

#define SYSTEM_STRIDE_PER_GRID_AXIS 8 
#define SYSTEM_STRIDE_PER_GRID (SYSTEM_STRIDE_PER_GRID_AXIS * SYSTEM_STRIDE_PER_GRID_AXIS)
class Grid {
public:
    Array2D<vec2> velocity;
    Array2D<vec2> velocity_backup;
    Array2D<vec4> boundary_normal;
    std::vector<Matrix2> _system;
    Array2D<real> mass;
    Array2D<int> id;
    std::vector<ivec2> id_to_pos;
    std::vector<ivec2> z_to_xy;
    int width, height;
    int valid_count;
    bool need_system;
    void initialize(int width, int height, bool need_system) {
        this->width = width;
        this->height = height;
        velocity.initialize(width, height);
        boundary_normal.initialize(width, height);
        mass.initialize(width, height);
        id.initialize(width, height);
        this->need_system = need_system;
        if (need_system)
            _system = std::vector<Matrix2>((width)* (height)* SYSTEM_STRIDE_PER_GRID);
        id_to_pos = std::vector<ivec2>(width * height);
        z_to_xy = std::vector<ivec2>(width * height);
        initialize_z_order();
    }
    void initialize_z_order() {
        //// requires width, height to be POT
        //if (((width & (width - 1)) || (height & (height - 1))) == 0) {
        //    for (auto &ind : mass.get_region()) {
        //        int z = ind.i * height + ind.j;
        //        int x = 0, y = 0;
        //        for (int s = 1, ss = 1; s < dim; s <<= 1, ss <<= 2) {
        //            if (z & ss) {
        //                x += s;
        //            }
        //            if (z & (ss << 1)) {
        //                y += s;
        //            }
        //        }
        //        z_to_xy[z] = ivec2(x, y);
        //    }

        //}
        for (auto &ind : mass.get_region()) {
            int z = ind.i * height + ind.j;
            z_to_xy[z] = ivec2(ind.i, ind.j);
        }
    }
    void backup_velocity() {
        velocity_backup = velocity;
    }
    void reset() {
        velocity = Vector2(0);
        mass = 0;
        if (need_system)
            memset(&_system[0], 0, _system.size() * sizeof(Matrix2));
    }
    int get_system_index(int i, int j, int k, int l) const {
        return (i * height + j) * SYSTEM_STRIDE_PER_GRID + (k - i + 3) * SYSTEM_STRIDE_PER_GRID_AXIS + (l - j + 3);
    }
    const Matrix2 system(int i, int j, int k, int l) const {
        return _system[get_system_index(i, j, k, l)];
    }
    Matrix2 &system(int i, int j, int k, int l) {
        return _system[get_system_index(i, j, k, l)];
    }
    void normalize_velocity() {
        for (auto &ind : velocity.get_region()) {
            if (mass[ind] > 0) { // Do not use EPS here!!
                velocity[ind] /= mass[ind];
            }
            else {
                velocity[ind] = Vector2(0, 0);
            }
            CV(velocity[ind]);
        }
    }
    void apply_external_force(vec2 acc, real delta_t) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if (mass[i][j] > 0) // Do not use EPS here!!
                    velocity[i][j] += acc * delta_t;
            }
        }
    }
    void apply_boundary_conditions(const LevelSet2D &levelset);
    void reorder_grids() {
        valid_count = 0;
        for (int z = 0; z < width * height; z++) {
            int i = z_to_xy[z].x, j = z_to_xy[z].y;
            if (mass[i][j] > 0) {
                id[i][j] = valid_count;
                id_to_pos[valid_count] = ivec2(i, j);
                valid_count += 1;
            }
            else {
                id[i][j] = -1;
            }
        }
        // printf("Reorder grids is disabled.\n"); // TODO: enable
        return;
        //for (int z = 0; z < dim * dim; z++) {
        //    int i = z_to_xy[z].x, j = z_to_xy[z].y;
        //    if (mass[i][j] > 0 && scene.image(i, j).r != 0.0f) {
        //        id[i][j] = valid_count;
        //        id_to_pos[valid_count] = ivec2(i, j);
        //        valid_count += 1;
        //    }
        //    else {
        //        id[i][j] = -1;
        //    }
        //}
    }
    void check_velocity() {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                if (!is_normal(velocity[i][j])) {
                    printf("Grid Velocity Check Fail!\n");
                    Pp(i);
                    Pp(j);
                    Pp(velocity[i][j]);
                    assert(false);
                }
            }
        }
    }
};


TC_NAMESPACE_END

