/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/system/threading.h>
#include <taichi/dynamics/poisson_solver3d.h>
#include <taichi/math/stencils.h>

TC_NAMESPACE_BEGIN

void PoissonSolver3D::initialize(const Config &config) {
    maximum_iterations = config.get_int("maximum_iterations");
}

// Maybe we are going to need Algebraic Multigrid in the future,
// but let's have a GMG with different boundary conditions support first...
// TODO: AMG, cache

class MultigridPoissonSolver3D : public PoissonSolver3D {
public:
    int max_level;
    Vector3i res;
    std::vector<Array> pressures, residuals, tmp_residuals;
    std::vector<BCArray> boundaries;
    const int size_threshold = 64;
    int num_threads;
    CellType padding;
    bool has_null_space;
    bool use_as_preconditioner;

    struct SystemRow {
        real inv_numerator;
        int neighbours;
        SystemRow(int _=0) { // _ is for Array3D initialization... Let's fix it later...
            inv_numerator = 0.0f;
            neighbours = 0;
        }
        CellType get_neighbour_cell_type(int k) const {
            return CellType((neighbours >> (k * 2)) % 4);
        }
        void set_neighbour_cell_type(int k, CellType c) {
            neighbours = ((neighbours & (~(3 << (2 * k)))) | (c << (2 * k)));
        }
        void print() {
            printf("%f", inv_numerator);
            for (int i = 0; i < 6; i++) {
                printf(" %d", get_neighbour_cell_type(i));
            }
            printf("\n");
        }

        static void test() {
            int neighbours[6] = {0};
            SystemRow r;
            for (int i = 0; i < 100000; i++) {
                int k = int(rand() * 6);
                int c = int(rand() * 3);
                if (rand() < 0.5f) {
                    neighbours[k] = c;
                    r.set_neighbour_cell_type(k, c);
                } else {
                    assert_info(neighbours[k] == r.get_neighbour_cell_type(k), "Test failed");
                }
            }
            printf("SystemRow tested.\n");
        }
    };

    typedef Array3D<SystemRow> System;

    bool test() const override {
        SystemRow::test();
        return true;
    }

    std::vector<System> systems;

    void set_boundary_condition(const BCArray &boundary) override {
        Vector3i res = this->res;
        boundaries.clear();
        boundaries.push_back(BCArray(res));
        // Iff we pad with Neumann and there's no dirichlet...
        has_null_space = padding == NEUMANN;

        for (auto &ind :boundary.get_region()) {
            if (boundary[ind] == DIRICHLET)
                has_null_space = false;
        }

        if (has_null_space) {
            // Let's remove the null space in an ad-hoc manner...
            // TODO: seperated components?
            // boundaries[0][0][0][0] = DIRICHLET;
            // error("null space detected");
        }

        // Step 1: figure out cell types
        for (int l = 0; l < max_level - 1; l++) {
            res /= 2;
            boundaries.push_back(BCArray(res));
            for (auto &ind : boundaries.back().get_region()) {
                auto &previous_boundary = boundaries[(int)boundaries.size() - 2];
                bool has_dirichlet = false;
                bool all_neumann = true;
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        for (int k = 0; k < 2; k++) {
                            char bc = previous_boundary[ind.i * 2 + i][ind.j * 2 + j][ind.k * 2 + k];
                            if (bc == DIRICHLET) {
                                has_dirichlet = true;
                                break;
                            }
                            if (bc != NEUMANN) {
                                all_neumann = false;
                            }
                        }
                    }
                }
                CellType bc = has_dirichlet ? DIRICHLET : (all_neumann ? NEUMANN : INTERIOR);
                boundaries.back()[ind] = bc;
            }
        }

        systems.clear();
        res = this->res;
        // Step 2: build the compressed systems
        for (int l = 0; l < max_level; l++) {
            System system(res);
            for (auto &ind: system.get_region()) {
                for (int i = 0; i < 6; i++) {
                    auto n_ind = ind + neighbour6_3d[i];
                    CellType cell;
                    if (boundaries[l].inside(n_ind)) {
                        cell = boundaries[l][n_ind];
                    } else {
                        cell = padding;
                    }
                    system[ind].set_neighbour_cell_type(i, cell);
                    if (cell == DIRICHLET || cell == INTERIOR) {
                        system[ind].inv_numerator += 1.0f;
                    }
                }
                if (boundaries[l][ind] != INTERIOR)
                    system[ind].inv_numerator = 0;
                else {
                    system[ind].inv_numerator = 1.0f / system[ind].inv_numerator;
                }
            }
            systems.push_back(system);
            res /= 2;
        }
    }

    void initialize(const Config &config) override {
        PoissonSolver3D::initialize(config);
        this->res = config.get_vec3i("res");
        this->num_threads = config.get_int("num_threads");
        auto padding_name = config.get_string("padding");
        use_as_preconditioner = false;
        assert_info(padding_name == "dirichlet" || padding_name == "neumann",
                    "'padding' has to be 'dirichlet' or 'neumann' instead of " + std::string(padding_name));
        if (padding_name == "dirichlet") {
            padding = DIRICHLET;
        } else {
            padding = NEUMANN;
        };
        this->max_level = 0;
        auto res = this->res;
        do {
            pressures.push_back(Array(res));
            residuals.push_back(Array(res));
            tmp_residuals.push_back(Array(res));
            assert_info(res[0] % 2 == 0, "odd width");
            assert_info(res[1] % 2 == 0, "odd height");
            assert_info(res[2] % 2 == 0, "odd depth");
            res /= 2;
            max_level++;
        } while (res[0] * res[1] * res[2] * 8 >= size_threshold);
    }

    void parallel_for_each_cell(Array &arr, int threshold, const std::function<void(const Index3D &index)> &func) {
        int max_side = std::max(std::max(arr.get_width(), arr.get_height()), arr.get_depth());
        int num_threads;
        if (max_side >= threshold) {
            num_threads = this->num_threads;
        }
        else {
            num_threads = 1;
        }
        ThreadedTaskManager::run(arr.get_width(), num_threads, [&](int x) {
            const int height = arr.get_height();
            const int depth = arr.get_depth();
            for (int y = 0; y < height; y++) {
                for (int z = 0; z < depth; z++) {
                    func(Index3D(x, y, z));
                }
            }
        });
    }

    bool get_has_null_space() {
        return has_null_space;
    }

    void gauss_seidel(const System &system, const Array &residual, Array &pressure, int rounds) {
        for (int i = 0; i < rounds; i++) {
            for (int c = 0; c < 2; c++) {
                parallel_for_each_cell(pressure, 128, [&](const Index3D &ind) {
                    int sum = ind.i + ind.j + ind.k;
                    if ((sum) % 2 == c) {
                        if (system[ind].inv_numerator > 0) {
                            real res = residual[ind];
                            for (int k = 0; k < 6; k++) {
                                Vector3i offset = neighbour6_3d[k];
                                if (system[ind].get_neighbour_cell_type(k) == INTERIOR) {
                                    res += pressure[ind + offset];
                                }
                            }
                            pressure[ind] = res * system[ind].inv_numerator;
                        } else {
                            pressure[ind] = 0.0f;
                        }
                    }
                });
            }
        }
    }

    void damped_jacobi(const System &system, const Array &residual, Array &pressure, int rounds) {
        for (int i = 0; i < rounds; i++) {
            for (int c = 0; c < 2; c++) {
                parallel_for_each_cell(pressure, 128, [&](const Index3D &ind) {
                    int sum = ind.i + ind.j + ind.k;
                    if ((sum) % 2 == c) {
                        if (system[ind].inv_numerator > 0) {
                            real res = residual[ind];
                            for (int k = 0; k < 6; k++) {
                                Vector3i offset = neighbour6_3d[k];
                                if (system[ind].get_neighbour_cell_type(k) == INTERIOR) {
                                    res += pressure[ind + offset];
                                }
                            }
                            real delta = res * system[ind].inv_numerator - pressure[ind];
                            pressure[ind] += delta * 0.666666666667f;
                        } else {
                            pressure[ind] = 0.0f;
                        }
                    }
                });
            }
        }
    }

    void apply_L(const System &system, const Array &pressure, Array &output) {
        for (auto &ind : pressure.get_region()) {
            if (system[ind].inv_numerator == 0.0f) {
                output[ind] = 0.0f;
                continue;
            }
            real pressure_center = pressure[ind];
            real res = 0.0f;
            for (int k = 0; k < 6; k++) {
                Vector3i offset = neighbour6_3d[k];
                CellType type = system[ind].get_neighbour_cell_type(k);
                if (type == INTERIOR) {
                    res += pressure_center - pressure[ind + offset];
                } else if (type == DIRICHLET) {
                    res += pressure_center;
                }
            }
            output[ind] = res;
        }
    }

    void compute_residual(const System &system, const Array &pressure, const Array &div, Array &residual) {
        parallel_for_each_cell(residual, 128, [&](const Index3D &ind) {
            if (system[ind].inv_numerator == 0) {
                residual[ind] = 0.0f;
                return;
            }
            real pressure_center = pressure[ind];
            real res = 0.0f;
            for (int k = 0; k < 6; k++) {
                Vector3i offset = neighbour6_3d[k];
                CellType type = system[ind].get_neighbour_cell_type(k);
                if (type == INTERIOR) {
                    res += pressure_center - pressure[ind + offset];
                } else if (type == DIRICHLET) {
                    res += pressure_center;
                }
            }
            residual[ind] = div[ind] - res;
        });
    }

    void downsample(const System &system, const Array &x, Array &x_downsampled) { // Restriction
        for (auto &ind : x_downsampled.get_region()) {
            if (system[ind].inv_numerator > 0) {
                x_downsampled[ind] =
                    x[ind.i * 2 + 0][ind.j * 2 + 0][ind.k * 2 + 0] +
                    x[ind.i * 2 + 0][ind.j * 2 + 0][ind.k * 2 + 1] +
                    x[ind.i * 2 + 0][ind.j * 2 + 1][ind.k * 2 + 0] +
                    x[ind.i * 2 + 0][ind.j * 2 + 1][ind.k * 2 + 1] +
                    x[ind.i * 2 + 1][ind.j * 2 + 0][ind.k * 2 + 0] +
                    x[ind.i * 2 + 1][ind.j * 2 + 0][ind.k * 2 + 1] +
                    x[ind.i * 2 + 1][ind.j * 2 + 1][ind.k * 2 + 0] +
                    x[ind.i * 2 + 1][ind.j * 2 + 1][ind.k * 2 + 1];
            } else {
                x_downsampled[ind] = 0.0f;
            }
        }
    }

    void prolongate(const System &system, Array &x, const Array &x_delta) const {
        for (auto &ind : x.get_region()) {
            // Do not prolongate to cells without a degree of freedom
            if (system[ind].inv_numerator > 0) {
                x[ind] += x_delta[ind.i / 2][ind.j / 2][ind.k / 2] * 0.5f;
            }
        }
    }

    void run(int level) {
        if (use_as_preconditioner)
            pressures[level].reset(0.0f);
        if (residuals[level].get_size() <= size_threshold) { // 4 * 4 * 4
            gauss_seidel(systems[level], residuals[level], pressures[level], 100);
        }
        else {
            gauss_seidel(systems[level], residuals[level], pressures[level], 4);
            {
                compute_residual(systems[level], pressures[level], residuals[level], tmp_residuals[level]);
                downsample(systems[level + 1], tmp_residuals[level], residuals[level + 1]);
                run(level + 1);
                prolongate(systems[level], pressures[level], pressures[level + 1]);
            }
            gauss_seidel(systems[level], residuals[level], pressures[level], 4);
        }
    }

    virtual void run(const Array &residual, Array &pressure, real pressure_tolerance) override {
        pressures[0] = pressure;
        residuals[0] = residual;
        int iterations = 0;
        do {
            iterations++;
            run(0);
            compute_residual(systems[0], pressures[0], residuals[0], tmp_residuals[0]);
            P(iterations);
            P(tmp_residuals[0].abs_max());
        } while (tmp_residuals[0].abs_max() > pressure_tolerance);
        pressure = pressures[0];
    }
};

class CGPoissonSolver3D : public MultigridPoissonSolver3D {
public:
    void initialize(const Config &config) {
        MultigridPoissonSolver3D::initialize(config);
    }
    Array apply_preconditioner(Array &r) {
        return r;
    }
    virtual void run(const Array &residual, Array &pressure, real pressure_tolerance) {
        pressure = 0;
        Array r(res), mu(res), tmp(res);
        mu = has_null_space ? r.get_average() : 0;
        r = residual - mu; //TODO: r = r - Lx
        double nu = r.abs_max();
        if (nu < pressure_tolerance)
            return;
        Array p = apply_preconditioner(r);
        double rho = p.dot_double(r);
        Array z(res);
        for (int count = 0; count <= maximum_iterations; count++) {
            apply_L(systems[0], p, z);
            double sigma = p.dot_double(z);
            double alpha = rho / max(1e-20, sigma);
            r.add_in_place(-(real)alpha, z);
            mu = has_null_space ? r.get_average() : 0.0f;
            r -= mu;
            nu = r.abs_max();
            r.print_abs_max_pos();
            printf(" CG iteration #%02d, nu=%f\n", count, nu);
            if (nu < pressure_tolerance || count == maximum_iterations) {
                pressure.add_in_place((real)alpha, p);
                return;
            }
            z = apply_preconditioner(r);
            double rho_new = z.dot_double(r);
            double beta = rho_new / rho;
            rho = rho_new;
            pressure.add_in_place((real)alpha, p);
            p = z.add((real)beta, p);
        }
    }
};

class MultigridPCGPoissonSolver3D : public MultigridPoissonSolver3D {
public:
    void initialize(const Config &config) {
        MultigridPoissonSolver3D::initialize(config);
        use_as_preconditioner = true;
    }
    Array apply_preconditioner(Array &r) {
        pressures[0] = 0;
        residuals[0] = r;
        MultigridPoissonSolver3D::run(0);
        return pressures[0];
    }
    virtual void run(const Array &residual, Array &pressure, real pressure_tolerance) {
        P(residual.sum());
        pressure = 0;
        Array r(res), mu(res), tmp(res);
        mu = has_null_space ? r.get_average() : 0;
        r = residual - mu; //TODO: r = r - Lx
        double nu = r.abs_max();
        if (nu < pressure_tolerance)
            return;
        Array p = apply_preconditioner(r);
        double rho = p.dot_double(r);
        Array z(res);
        for (int count = 0; count <= maximum_iterations; count++) {
            apply_L(systems[0], p, z);
            double sigma = p.dot_double(z);
            double alpha = rho / max(1e-20, sigma);
            r.add_in_place(-(real)alpha, z);
            mu = has_null_space ? r.get_average() : 0.0f;
            r -= mu;
            nu = r.abs_max();
            r.print_abs_max_pos();
            printf(" MGPCG iteration #%02d, nu=%f\n", count, nu);
            if (nu < pressure_tolerance || count == maximum_iterations) {
                pressure.add_in_place((real)alpha, p);
                return;
            }
            z = apply_preconditioner(r);
            double rho_new = z.dot_double(r);
            double beta = rho_new / rho;
            rho = rho_new;
            pressure.add_in_place((real)alpha, p);
            p = z.add((real)beta, p);
        }
    }
};

TC_IMPLEMENTATION(PoissonSolver3D, MultigridPoissonSolver3D, "mg");
TC_IMPLEMENTATION(PoissonSolver3D, CGPoissonSolver3D, "cg");
TC_IMPLEMENTATION(PoissonSolver3D, MultigridPCGPoissonSolver3D, "mgpcg");

TC_NAMESPACE_END
