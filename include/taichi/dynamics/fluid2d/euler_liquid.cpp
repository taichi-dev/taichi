/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include "euler_liquid.h"

TC_NAMESPACE_BEGIN

real volume_control_i = 0.0f;
real volume_control_p = 0.0f;

void EulerLiquid::set_levelset(const LevelSet2D & boundary_levelset)
{
    this->boundary_levelset = boundary_levelset;
    //rebuild_levelset(this->boundary_levelset, levelset_band + 1);
}

void EulerLiquid::initialize(const Config &config) {
    initialize_solver(config);
    levelset_band = config.get_real("levelset_band");
}

void EulerLiquid::initialize_solver(const Config &config)
{
    width = config.get("simulation_width", 64);
    height = config.get("simulation_height", 64);
    kernel_size = config.get("kernel_size", 1);
    cfl = config.get("cfl", 0.1f);
    u = Array(width + 1, height, 0.0f, Vector2(0.0f, 0.5f));
    u_weight = Array(width + 1, height, 0.0f, Vector2(0.0f, 0.5f));
    v = Array(width, height + 1, 0.0f, Vector2(0.5f, 0.0f));
    v_weight = Array(width, height + 1, 0.0f, Vector2(0.5f, 0.0f));
    cell_types = Array2D<CellType>(width, height, CellType::AIR, Vector2(0.5f, 0.5f));
    gravity = config.get_vec2("gravity");
    maximum_iterations = config.get("maximum_iterations", 300);
    tolerance = config.get("tolerance", 1e-4f);
    theta_threshold = config.get("theta_threshold", 0.1f);
    initialize_pressure_solver();
    liquid_levelset.initialize(width, height, Vector2(0.5f, 0.5f));
    t = 0;
}

Vector2 EulerLiquid::sample_velocity(Vector2 position, const Array &u, const Array &v) {
    if (kernel_size == 1) {
        return Vector2(u.sample(position), v.sample(position));
    }
    else {
        real inv_kernel_size = 1.0f / kernel_size;
        int extent = (kernel_size + 1) / 2;
        int x, y;
        real tot_weight, tot;

        x = (int)floor(position.x);
        y = (int)floor(position.y - 0.5);
        tot_weight = 0.0f; tot = 0.0f;
        for (int dx = -extent + 1; dx <= extent; dx++) {
            for (int dy = -extent + 1; dy <= extent; dy++) {
                int nx = x + dx, ny = y + dy;
                if (!u.inside(nx, ny))
                    continue;
                real weight = kernel(inv_kernel_size * (position - Vector2(nx, ny + 0.5f)));
                tot += u[nx][ny] * weight;
                tot_weight += weight;
            }
        }
        real vx = tot / tot_weight;

        x = (int)floor(position.x - 0.5);
        y = (int)floor(position.y);
        tot_weight = 0.0f; tot = 0.0f;
        for (int dx = -extent + 1; dx <= extent; dx++) {
            for (int dy = -extent + 1; dy <= extent; dy++) {
                int nx = x + dx, ny = y + dy;
                if (!v.inside(nx, ny)) {
                    continue;
                }
                real weight = kernel(inv_kernel_size * (position - Vector2(nx + 0.5f, ny)));
                tot += v[nx][ny] * weight;
                tot_weight += weight;
            }
        }

        real vy = tot / tot_weight;
        return Vector2(vx, vy);
    }
}

Vector2 EulerLiquid::sample_velocity(Vector2 position) {
    return sample_velocity(position, u, v);
}

bool EulerLiquid::check_u_activity(int i, int j) {
    if (i < 1 || j < 0 || i >= width || j >= height) return false;
    return cell_types[i - 1][j] == CellType::WATER ||
        cell_types[i][j] == CellType::WATER;
}

bool EulerLiquid::check_v_activity(int i, int j) {
    if (i < 0 || j < 1 || i >= width || j >= height) return false;
    return cell_types[i][j - 1] == CellType::WATER ||
        cell_types[i][j] == CellType::WATER;
}

void EulerLiquid::level_set_extrapolate() {
    //for (int i = 0; i < width + 1; i++) {
    //    for (int j = 0; j < height; j++) {
    //        Vector2 p = Vector2(i, j + 0.5);
    //        if (level_set->sample(p) > 0) {
    //            Vector2 q = level_set->sample_closest(p);
    //            u[i][j] = sample_velocity(q).x;
    //        }
    //    }
    //}
    //for (int i = 0; i < width; i++) {
    //    for (int j = 0; j < height + 1; j++) {
    //        Vector2 p = Vector2(i + 0.5f, j);
    //        if (level_set->sample(p) > 0) {
    //            Vector2 q = level_set->sample_closest(p);
    //            v[i][j] = sample_velocity(q).y;
    //        }
    //    }
    //}

}

void EulerLiquid::simple_extrapolate() {
    const int dx[4]{ 1, -1, 0, 0 };
    const int dy[4]{ 0, 0, 1, -1 };
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (check_u_activity(i, j)) continue;
            real sum = 0.0f, num = 0.0f;
            for (int k = 0; k < 4; k++) {
                int nx = i + dx[k], ny = j + dy[k];
                if (check_u_activity(nx, ny)) {
                    num += 1.0f;
                    sum += u[nx][ny];
                }
            }
            if (num == 0.0f)
                u[i][j] = 0.0f;
            else
                u[i][j] = sum / num;
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 1; j < height; j++) {
            if (check_v_activity(i, j)) continue;
            real sum = 0.0f, num = 0.0f;
            for (int k = 0; k < 4; k++) {
                int nx = i + dx[k], ny = j + dy[k];
                if (check_v_activity(nx, ny)) {
                    num += 1.0f;
                    sum += v[nx][ny];
                }
            }
            if (num == 0.0f)
                v[i][j] = 0.0f;
            else {
                v[i][j] = sum / num;
            }
        }
    }
}

void EulerLiquid::step(real delta_t)
{
    real simulation_time = 0.0f;
    while (simulation_time < delta_t - eps) {
        real purpose_dt = get_dt_with_cfl_1() * cfl;
        real thres = 0.001f;
        if (purpose_dt < delta_t * thres) {
            purpose_dt = delta_t * thres;
            printf("substep dt too small, clamp.\n");
            Particle fastest;
            real avg = 0;
            for (auto &p : particles) {
                if (length(p.velocity) > length(fastest.velocity)) {
                    fastest = p;
                }
                avg += abs(p.velocity.x) + abs(p.velocity.y);
            }
            printf("Fastest particle:\n");
            P(fastest.position);
            P(fastest.velocity);
            avg /= particles.size() * 2;
            P(avg);
        }
        real dt = std::min(delta_t - simulation_time, purpose_dt);
        substep(dt);
        simulation_time += dt;
    }
    //compute_liquid_levelset();
}

void EulerLiquid::compute_liquid_levelset()
{
    error("error");
    liquid_levelset.reset(1e7f); // Do not use INF here, otherwise interpolation will get NAN...
    for (auto &p : particles) {
        for (auto &ind : liquid_levelset.get_rasterization_region(p.position, 3)) {
            Vector2 delta_pos = ind.get_pos() - p.position;
            liquid_levelset[ind] = std::min(liquid_levelset[ind], length(delta_pos) - p.radius);
        }
    }
    for (auto &ind : liquid_levelset.get_region()) {
        if (liquid_levelset[ind] < 0.5f) {
            if (boundary_levelset.sample(ind.get_pos()) < 0)
                liquid_levelset[ind] = -0.5f;
        }
    }
}


void EulerLiquid::advect_liquid_levelset(real delta_t) {
    Array old = liquid_levelset;
    for (auto &ind : liquid_levelset.get_region()) {
        liquid_levelset[ind] = old.sample(ind.get_pos() - delta_t * sample_velocity(ind.get_pos(), u, v));
    }
    rebuild_levelset(liquid_levelset, levelset_band + 1);
}

void EulerLiquid::rebuild_levelset(LevelSet2D &levelset, real band) {
    // Actually, we use a brute-force initialization here
    Array old = levelset;
    levelset.reset(band);
    auto update = [&](const Index2D &a, const Index2D &b) {
        real phi_0 = old[a], phi_1 = old[b];
        if (phi_0 * phi_1 > 0) {
            return;
        }
        // Free surface detected
        real p = std::abs(phi_0 / (phi_1 - phi_0));
        Vector2 pos = lerp(p, Vector2(a.i, a.j), Vector2(b.i, b.j));
        for (int i = std::max(0, int(floor(a.i - band)));
             i <= std::min(levelset.get_width() - 1, int(b.i + band)); i++) {
            for (int j = std::max(0, int(floor(a.j - band)));
                 j <= std::min(levelset.get_height() - 1, int(b.j + band)); j++) {
                real l = length(Vector2(i, j) - pos);
                levelset[i][j] = std::min(levelset[i][j], l);
            }
        }
    };
    for (auto &ind : old.get_region()) {
        if (ind.i > 0) {
            const Index2D a = ind.neighbour(-1, 0);
            const Index2D b = ind;
            update(a, b);
        }
        if (ind.j > 0) {
            const Index2D a = ind.neighbour(0, -1);
            const Index2D b = ind;
            update(a, b);
        }
    }
    for (auto &ind : levelset.get_region()) {
        levelset[ind] *= sgn(old[ind]);
    }
}


Array EulerLiquid::advect(const Array & arr, real delta_t)
{
    Array arr_out(arr.get_width(), arr.get_height(), 0, arr.get_storage_offset());
    for (auto &ind : arr.get_region()) {
        Vector2 position = ind.get_pos();
        Vector2 velocity = sample_velocity(position);
        velocity = sample_velocity(position - delta_t * 0.5f * velocity);
        arr_out[ind] = arr.sample(position - delta_t * velocity);
    }
    return arr_out;
}

bool EulerLiquid::check_diag_domination()
{
    for (auto &ind : Ad.get_region()) {
        real res = Ad[ind];
        res -= abs(Ax[ind]);
        res -= abs(Ay[ind]);
        if (ind.i > 0) {
            res -= abs(Ax[ind + Vector2(-1, 0)]);
        }
        if (ind.j > 0) {
            res -= abs(Ay[ind + Vector2(0, -1)]);
        }
        if (res < -1e-7f) {
            return false;
        }
    }
    return true;
}


void EulerLiquid::advect(real delta_t) {
    real total_energy = 0;
    /*
    for (auto &particle : particles) {
        Vector2 velocity = sample_velocity(particle.position);
        if (true) {
            Particle mid = particle;
            mid.move(delta_t * 0.5f * velocity);
            mid.position = clamp_particle_position(mid.position);
            velocity = sample_velocity(mid.position);
            particle.move(delta_t * velocity);
            particle.position = clamp_particle_position(particle.position);
        }
        else {
            particle.move(delta_t * velocity);
            particle.position = clamp_particle_position(particle.position);
        }
        total_energy += dot(velocity, velocity) * 0.5f;
        //total_energy -= glm::dot(particle.position, gravity);
    }
    */
    Array new_u = advect(u, delta_t), new_v = advect(v, delta_t);
    u = new_u;
    v = new_v;
}

void EulerLiquid::apply_external_forces(real delta_t) {
    for (int i = 1; i < width; i++) {
        for (int j = 0; j < height; j++) {
            u[i][j] += gravity.x * delta_t;
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 1; j < height; j++) {
            v[i][j] += gravity.y * delta_t;
        }
    }
}

Vector2 EulerLiquid::position_noise()
{
    return Vector2(rand() - 0.5f, rand() - 0.5f);
}

bool EulerLiquid::inside(int x, int y) {
    return 0 <= x && x < width && 0 <= y && y < height;
}

void EulerLiquid::update_velocity_weights() {
    for (auto &ind : u.get_region()) {
        u_weight[ind] = LevelSet2D::fraction_outside(boundary_levelset[ind], boundary_levelset[ind.neighbour(Vector2i(0, 1))]);
    }
    for (auto &ind : v.get_region()) {
        v_weight[ind] = LevelSet2D::fraction_outside(boundary_levelset[ind], boundary_levelset[ind.neighbour(Vector2i(1, 0))]);
    }
}

void EulerLiquid::prepare_for_pressure_solve() {
    Ax = 0;
    Ay = 0;
    Ad = 0;
    E = 0;
    particles.clear();
    const real theta_threshold = 0.01f;
    Array2D<char> boundary_cell(width, height, false);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            int fluid_corner = 0, boundary_corner = 0;
            fluid_corner += liquid_levelset.sample(Vector2(i + 0, j + 0)) < 0;
            fluid_corner += liquid_levelset.sample(Vector2(i + 0, j + 1)) < 0;
            fluid_corner += liquid_levelset.sample(Vector2(i + 1, j + 0)) < 0;
            fluid_corner += liquid_levelset.sample(Vector2(i + 1, j + 1)) < 0;
            boundary_corner += boundary_levelset[i + 0][j + 0] < 0;
            boundary_corner += boundary_levelset[i + 0][j + 1] < 0;
            boundary_corner += boundary_levelset[i + 1][j + 0] < 0;
            boundary_corner += boundary_levelset[i + 1][j + 1] < 0;
            if (fluid_corner > 0 && boundary_corner > 0) {
                boundary_cell[i][j] = true;
            }
        }
    }
    for (auto &ind : cell_types.get_region()) {
        int i = ind.i, j = ind.j;
        real phi = liquid_levelset[ind];
        if (phi >= 0) {
            if (!boundary_cell[i][j]) {
                continue;
            }
        } else {
        }
        real lhs = 0;
        real neighbour_phi;
        real vel_weight;

        neighbour_phi = liquid_levelset.sample(ind.get_pos() - Vector2(1, 0));
        if (neighbour_phi < 0 || boundary_cell[i][j] || boundary_cell[i - 1][j]) {
            vel_weight = u_weight[ind];
            lhs += vel_weight;
        }
        else {
            real theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += vel_weight / theta;
        }

        neighbour_phi = liquid_levelset.sample(ind.get_pos() + Vector2(1, 0));
        if (neighbour_phi < 0 || boundary_cell[i][j] || boundary_cell[i + 1][j]) {
            vel_weight = u_weight[ind.neighbour(Vector2i(1, 0))];
            lhs += vel_weight;
            Ax[i][j] -= vel_weight;
        }
        else {
            real theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += vel_weight / theta;
        }

        neighbour_phi = liquid_levelset.sample(ind.get_pos() - Vector2(0, 1));
        if (neighbour_phi < 0 || boundary_cell[i][j] || boundary_cell[i][j - 1]) {
            vel_weight = v_weight[ind];
            lhs += vel_weight;
        }
        else {
            real theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += vel_weight / theta;
        }


        neighbour_phi = liquid_levelset.sample(ind.get_pos() + Vector2(0, 1));
        if (neighbour_phi < 0 || boundary_cell[i][j] || boundary_cell[i][j + 1]) {
            vel_weight = v_weight[ind.neighbour(Vector2i(0, 1))];
            lhs += vel_weight;
            Ay[i][j] -= vel_weight;
        }
        else {
            real theta = max(theta_threshold, LevelSet2D::fraction_inside(phi, neighbour_phi));
            lhs += vel_weight / theta;
        }

        Ad[ind] = lhs;
    }

    if (!check_diag_domination()) {
        printf("Warning: Non diagonally dominant matrix found!\n");
    }

    real tao = 0.97f, sigma = 0.25f;

    for (auto &ind : cell_types.get_region()) {
        if (Ad[ind] > 0) {
            real e = Ad[ind];
            real e_tao = 0.0f;
            Index2D nei;
            nei = ind.neighbour(Vector2i(-1, 0));
            if (cell_types.inside(nei) && Ad[nei] > 0) {
                e -= sqr(Ax[nei] * E[nei]);
                e_tao -= Ax[nei] * Ay[nei] * sqr(E[nei]);
            }
            nei = ind.neighbour(Vector2i(0, -1));
            if (cell_types.inside(nei) && Ad[nei] > 0) {
                e -= sqr(Ay[nei] * E[nei]);
                e_tao -= Ay[nei] * Ax[nei] * sqr(E[nei]);
            }
            e += e_tao * tao;
            if (e < sigma * Ad[ind]) e = Ad[ind];
            assert_info(e >= 0, "Negative e!");
            E[ind] = 1.0f / sqrtf(e);
            if (!is_normal(E[ind])) {
                P(E[ind]);
                P(e);
                printf("Bad E.\n");
            }
        }
    }
}

Array EulerLiquid::apply_A(const Array &x) {
    Array y(width, height);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (Ad[i][j] > 0) {
                real t = 0;
                if (0 < i)
                    t += Ax[i - 1][j] * x[i - 1][j];
                if (i < width - 1)
                    t += Ax[i][j] * x[i + 1][j];
                if (0 < j)
                    t += Ay[i][j - 1] * x[i][j - 1];
                if (j < height - 1)
                    t += Ay[i][j] * x[i][j + 1];
                t += Ad[i][j] * x[i][j];
                y[i][j] = t;
            }
            else {
                y[i][j] = 0;
            }
        }
    }
    return y;
}


Array EulerLiquid::solve_pressure_naive() {
    static int total_count = 0;
    int count = 0;
    Array r = get_rhs(), z, s;
    z = apply_preconditioner(r);
    s = z;
#define CH(v) if (!v.is_normal()) printf("Abnormal value doring CG: %s [Ln %d]\n", #v, __LINE__);
    pressure = 0;
    double sigma = z.dot_double(r);
    double zs;

    for (count = 0; count < maximum_iterations; count++){
        z = apply_A(s);
        zs = z.dot_double(s);
        double alpha = sigma / max(1e-6, zs);
        pressure = pressure.add((real)alpha, s);
        r = r.add(-(real)alpha, z);
        if (r.abs_max() < tolerance) break;
        z = apply_preconditioner(r);
        double sigma_new = z.dot_double(r);
        double beta = sigma_new / sigma;
        s = z.add((real)beta, s);
        sigma = sigma_new;
    }
    total_count += count;
    printf("t = %f, iterated %d times, avg = %f\n", t, count, total_count / t);
    return pressure;
}

void EulerLiquid::project(real delta_t) {
    update_volume_controller();
    apply_boundary_condition();
    prepare_for_pressure_solve();
    p = solve_pressure_naive();
    if (!(p.is_normal())) {
        printf("Abnormal pressure!!!!!\n");
    }
    apply_pressure(p);
    apply_boundary_condition();
}

void EulerLiquid::mark_cells() {
    cell_types = CellType::AIR;
    for (auto &ind : cell_types.get_region()) {
        if (liquid_levelset.sample(ind.get_pos()) < 0) {
            cell_types[ind] = CellType::WATER;
        }
    }
    /*
    for (auto &particle : particles) {
        int x = (int)particle.position.x, y = (int)particle.position.y;
        cell_types[x][y] = CellType::WATER;
    }
    */
}

void EulerLiquid::substep(real delta_t) {
    u_weight.print("u_weight");
    v_weight.print("v_weight");
    rebuild_levelset(liquid_levelset, levelset_band);
    update_velocity_weights();
    apply_external_forces(delta_t);
    mark_cells();
    project(delta_t);
    simple_extrapolate();
    advect(delta_t);
    advect_liquid_levelset(delta_t);
    for (auto &ind : liquid_levelset.get_region())
        liquid_levelset[ind] = std::max(liquid_levelset[ind], -boundary_levelset.sample(ind.get_pos()));
    t += delta_t;
}

void EulerLiquid::apply_pressure(const Array &p) {
    for (int i = 0; i < width - 1; i++) {
        for (int j = 0; j < height; j++) {
            real theta = LevelSet2D::fraction_inside(liquid_levelset[i][j], liquid_levelset[i + 1][j]);
            if (u_weight[i + 1][j] > 0 && theta > 0)
                u[i + 1][j] += (p[i][j] - p[i + 1][j]) / std::max(theta_threshold, theta);
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height - 1; j++) {
            real theta = LevelSet2D::fraction_inside(liquid_levelset[i][j], liquid_levelset[i][j + 1]);
            if (v_weight[i][j + 1] > 0 && theta > 0)
                v[i][j + 1] += (p[i][j] - p[i][j + 1]) / std::max(theta_threshold, theta);
        }
    }
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (liquid_levelset[i][j] > 0) {
                continue;
            }
            real div = 0;
            div += u[i][j] * u_weight[i][j];
            div += v[i][j] * v_weight[i][j];
            div -= u[i + 1][j] * u_weight[i + 1][j];
            div -= v[i][j + 1] * v_weight[i][j + 1];
            if (abs(div) > 1e-3) {
                printf("%d  %d div %f\n", i, j, div);
            }
        }
    }
}

void EulerLiquid::apply_boundary_condition() {
    for (auto &ind : u.get_region()) {
        if (u_weight[ind] == 0.0f) {
            u[ind] = 0.0f;
        }
    }
    for (auto &ind : v.get_region()) {
        if (v_weight[ind] == 0.0f) {
            v[ind] = 0.0f;
        }
    }
}

real EulerLiquid::get_current_time() {
    return t;
}

void EulerLiquid::add_particle(Fluid::Particle & particle) {
    liquid_levelset[(int)floor(particle.position.x)][(int)floor(particle.position.y)] = -1;
    //particles.push_back(particle);
}

std::vector<Fluid::Particle> EulerLiquid::get_particles() {
    return particles;
}

LevelSet2D EulerLiquid::get_liquid_levelset() {
    return liquid_levelset;
}


Array EulerLiquid::apply_preconditioner(const Array &r) {
    q = 0;
    z = 0;
    assert_info(E.is_normal(), "Abnormal E!\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (Ad[i][j] > 0) {
                real t = r[i][j];
                if (i > 0)
                    t -= Ax[i - 1][j] * E[i - 1][j] * q[i - 1][j];
                if (j > 0)
                    t -= Ay[i][j - 1] * E[i][j - 1] * q[i][j - 1];
                q[i][j] = t * E[i][j];
            }
        }
    }
    for (int i = width - 1; i >= 0; i--) {
        for (int j = height - 1; j >= 0; j--) {
            if (Ad[i][j] > 0) {
                real t = q[i][j];
                if (i < width - 1) {
                    t -= Ax[i][j] * E[i][j] * z[i + 1][j];
                }
                if (j < height - 1) {
                    t -= Ay[i][j] * E[i][j] * z[i][j + 1];
                }
                z[i][j] = t * E[i][j];
            }
        }
    }
    return z;
}

Array EulerLiquid::get_rhs() {
    Array r(width, height, 0);
    real correction = get_volume_correction();
    for (auto &ind : cell_types.get_region()) {
        if (Ad[ind] > 0) {
            real rhs =
                - u[ind.neighbour(Vector2i(1, 0))] * u_weight[ind.neighbour(Vector2i(1, 0))]
                + u[ind] * u_weight[ind]
                - v[ind.neighbour(Vector2i(0, 1))] * v_weight[ind.neighbour(Vector2i(0, 1))]
                + v[ind] * v_weight[ind];
            r[ind] = rhs + correction;
        }
    }
    return r;
}

void EulerLiquid::apply_viscosity(real delta_t) {
    //static real tmp[2048][2048];
    //for (int i = 1; i < width; i++) {
    //    for (int j = 1; j < height - 1; j++) {
    //        real sum = u[i][j] * 4;
    //        sum -= u[i - 1][j];
    //        sum -= u[i + 1][j];
    //        sum -= u[i][j - 1];
    //        sum -= u[i][j + 1];
    //        tmp[i][j] = sum;
    //    }
    //}
    //for (int i = 1; i < width; i++) {
    //    for (int j = 1; j < height - 1; j++) {
    //        u[i][j] -= viscosity * delta_t * tmp[i][j];
    //    }
    //}
    //for (int i = 1; i < width - 1; i++) {
    //    for (int j = 1; j < height; j++) {
    //        real sum = v[i][j] * 4;
    //        sum -= v[i - 1][j];
    //        sum -= v[i + 1][j];
    //        sum -= v[i][j - 1];
    //        sum -= v[i][j + 1];
    //        tmp[i][j] = sum;
    //    }
    //}
    //for (int i = 1; i < width - 1; i++) {
    //    for (int j = 1; j < height; j++) {
    //        v[i][j] -= viscosity * delta_t * tmp[i][j];
    //    }
    //}

}

int EulerLiquid::count_water_cells() {
    int ret = 0;
    for (auto &cell : cell_types)
        ret += cell == CellType::WATER;
    return ret;
}

void EulerLiquid::initialize_volume_controller() {
    integrate_water_cells_difference = 0.0f;
    target_water_cells = (real)count_water_cells();
    last_water_cells = (real)count_water_cells();
}

void EulerLiquid::update_volume_controller() {
    int current_water_cells = count_water_cells();
    integrate_water_cells_difference +=
        current_water_cells - target_water_cells;
    last_water_cells = (real)current_water_cells;
    real factor = 1.0f / width / height;
    volume_correction_factor =
        -factor * (integrate_water_cells_difference * volume_control_i +
            (current_water_cells - target_water_cells) *
            volume_control_p);
}

real EulerLiquid::get_volume_correction() {
    //if (enable_volume_control)
    if (false)
        return volume_correction_factor;
    else
        return 0.0f;
}

void EulerLiquid::advect_level_set(real delta_t) {
    //real tmp[width][height];
    //for (int i = 0; i < width; i++) {
    //    for (int j = 0; j < height; j++) {
    //        Vector2 position = sl_position(level_set->get_location(i, j), delta_t);
    //        tmp[i][j] = level_set->sample(position);
    //    }
    //}
    //memcpy(level_set->signed_distance, tmp, sizeof(tmp));
}

Vector2 EulerLiquid::sl_position(Vector2 position, real delta_t) {
    Vector2 velocity = sample_velocity(position, u, v);
    Vector2 mid = clamp(position - velocity * 0.5f * delta_t);
    velocity = sample_velocity(mid, u, v);
    position = clamp(position - velocity * delta_t);
    return position;
}

void EulerLiquid::print_u() {
    printf("u:\n");
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i <= width; i++) {
            printf("%.4f ", u[i][j]);
        }
        printf("\n");
    }
}

void EulerLiquid::print_v() {
    printf("v:\n");
    for (int j = height; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            printf("%+6.4f ", v[i][j]);
        }
        printf("\n");
    }

}

void EulerLiquid::initialize_pressure_solver() {
    pressure = Array(width, height, 0.0);
    Ad = Array(width, height);
    Ax = Array(width, height);
    Ay = Array(width, height);
    E = Array(width, height);
    p = Array(width, height, 0.0f);
    q = Array(width, height);
    z = Array(width, height);
    water_cell_index = Array2D<int>(width, height);
}

Vector2 EulerLiquid::clamp_particle_position(Vector2 pos) {
    pos = Vector2(clamp(pos.x, 0.0f, (real)width), clamp(pos.y, 0.0f, (real)height));
    real phi = boundary_levelset.sample(pos);
    if (phi < 0) {
        pos -= boundary_levelset.get_normalized_gradient(pos) * phi;
    }
    return pos;
}

real EulerLiquid::get_dt_with_cfl_1()
{
    return 1 / max(get_max_grid_speed(), 1e-5f);
}

real EulerLiquid::get_max_grid_speed()
{
    real maximum_speed = 0;
    for (auto &vel : u)
        maximum_speed = max(abs(vel), maximum_speed);
    for (auto &vel : v)
        maximum_speed = max(abs(vel), maximum_speed);
    return maximum_speed;
}

Array EulerLiquid::get_density()
{
    return density;
}

void EulerLiquid::add_source(const Config & config)
{
    sources.push_back(config);
}

Array EulerLiquid::get_pressure()
{
    return pressure;
}

TC_IMPLEMENTATION(Fluid, EulerLiquid, "liquid");

TC_NAMESPACE_END
