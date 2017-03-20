/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/image/tone_mapper.h>
#include <taichi/math/array_op.h>
#include <taichi/physics/physics_constants.h>
#include <taichi/dynamics/poisson_solver2d.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(ToneMapper, "tone_mapper")

class GradientDomainTMO final : public ToneMapper {
protected:
    real pyramid_sigma;
    real alpha, beta;
    real s;
    int num_threads;
    int max_solver_iterations;
public:
    void initialize(const Config &config) override {
        pyramid_sigma = config.get("pyramid_sigma", 1.0f);
        num_threads = config.get("num_threads", 1);
        max_solver_iterations = config.get("max_solver_iterations", 100);
        alpha = config.get_real("alpha");
        beta = config.get_real("beta");
        s = config.get_real("s");
    }

    virtual Array2D<Vector3> apply(const Array2D<Vector3> &inp) override {
        int width = inp.get_width(), height = inp.get_height();
        assert_info(inp.get_width() == inp.get_height(), "only square image supported currently.");

        Array2D<real> lum(inp.get_width(), inp.get_height());
        Array2D<real> log_lum(inp.get_width(), inp.get_height());
        for (auto &ind : inp.get_region()) {
            lum[ind] = luminance(inp[ind]);
            log_lum[ind] = std::log(lum[ind] + 1e-4f);
        }
        std::vector<Array2D<real>> pyramid;
        std::vector<Array2D<real>> phi;
        Array2D<Vector2> G(width, height);
        Array2D<real> div_G(width, height);

        pyramid.push_back(log_lum);
        int size = inp.get_width();
        while (size > 32) {
            size /= 2;
            auto blurred = gaussian_blur(pyramid.back(), pyramid_sigma);
            pyramid.push_back(take_downsampled(blurred, 2));
        }
        phi.resize(pyramid.size());
        for (int k = (int)pyramid.size() - 1; k >= 0; k--) {
            phi[k] = Array2D<real>(pyramid[k].get_width(), pyramid[k].get_height());
            Array2D<real> grad_norm(pyramid[k].get_width(), pyramid[k].get_height());
            real scale = std::pow(0.5f, k + 1);
            for (auto &ind : grad_norm.get_region()) {
                real grad_x = 0, grad_y = 0;
                if (ind.i > 0) {
                    grad_x += pyramid[k][ind] - pyramid[k][ind.neighbour(-1, 0)];
                }
                if (ind.i < pyramid[k].get_width() - 1) {
                    grad_x += pyramid[k][ind.neighbour(1, 0)] - pyramid[k][ind];
                }
                if (ind.j > 0) {
                    grad_y += pyramid[k][ind] - pyramid[k][ind.neighbour(0, -1)];
                }
                if (ind.j < pyramid[k].get_height() - 1) {
                    grad_y += pyramid[k][ind.neighbour(0, 1)] - pyramid[k][ind];
                }
                grad_x *= 0.5;
                grad_y *= 0.5;
                grad_norm[ind] = std::hypot(grad_x, grad_y) * scale;
            }
            real avg = grad_norm.get_average();
            for (auto &ind : phi[k].get_region()) {
                real norm = std::max(grad_norm[ind], 1e-5f);
                phi[k][ind] = std::pow(norm / (alpha * avg), beta - 1);
                if (k != (int)pyramid.size() - 1) {
                    phi[k][ind] *= phi[k + 1].sample(ind.get_pos() * 2.0f);
                }
            }
        }
        auto oup = inp;
        for (auto &ind : G.get_region()) {
            real grad_x, grad_y;
            if (ind.i + 1 < width)
                grad_x = log_lum[ind.neighbour(1, 0)] - log_lum[ind];
            else
                grad_x = 0;
            if (ind.j + 1 < height)
                grad_y = log_lum[ind.neighbour(0, 1)] - log_lum[ind];
            else
                grad_y = 0;
            G[ind] = Vector2(grad_x, grad_y) * phi[0][ind];
        }
        for (auto &ind : div_G.get_region()) {
            real div_x = G[ind].x, div_y = G[ind].y;
            if (ind.i > 0)
                div_x -= G[ind.neighbour(-1, 0)].x;
            if (ind.j > 0)
                div_y -= G[ind.neighbour(0, -1)].y;
            div_G[ind] = -(div_x + div_y);
        }
        auto poisson_solver = create_instance<PoissonSolver2D>("mgpcg");
        Config cfg;
        cfg.set("res", Vector2i(width, height)).set("num_threads", num_threads).set("padding", "neumann").
            set("maximum_iterations", max_solver_iterations);
        poisson_solver->initialize(cfg);

        Array2D<PoissonSolver2D::CellType> boundary(width, height, PoissonSolver2D::INTERIOR);
        poisson_solver->set_boundary_condition(boundary);

        real avg_div_G = div_G.get_average();
        for (auto &ind : div_G.get_region()) {
            div_G[ind] -= avg_div_G;
        }

        Array2D<real> I(width, height);
        poisson_solver->run(div_G, I, 1e-5f);
        for (auto &ind : oup.get_region()) {
            for (int i = 0; i < 3; i++) {
                oup[ind][i] = std::pow(inp[ind][i] / (lum[ind] + 1e-30f), s) * std::exp(I[ind]);
            }
        }
        return oup;
    }
};

TC_IMPLEMENTATION(ToneMapper, GradientDomainTMO, "gradient")

TC_NAMESPACE_END