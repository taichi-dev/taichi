/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/linalg.h>
#include <taichi/math/array_2d.h>
#include "mpm_utils.h"
#include <taichi/common/util.h>
#include <taichi/math/levelset_2d.h>

TC_NAMESPACE_BEGIN

extern long long kernel_calc_counter;

struct MPMParticle {
    // Color for visualization: if x=-1, use default color from color scheme.
    Vector3 color = Vector3(-1, 0, 0);
    Vector2 pos, v;
    Vector2i mipos;
    Matrix2 dg_e, dg_p, tmp_force;
    real mass;
    real vol = -1.0f;
    real cache_w[16];
    Vector2 cache_gw[16];
    Vector2 cache_dg_gw[16];
    Matrix2 b;
    Matrix2 dg_cache;
    static long long instance_count;
    long long id = instance_count++;
    enum State {
        INACTIVE = 0,
        BUFFER = 1,
        UPDATING = 2,
    };
    int state = INACTIVE;
    int march_interval;

    MPMParticle() {
        dg_e = Matrix2(1.0f);
        dg_p = Matrix2(1.0f);
        b = Matrix2(0);
    }

    // Maximum dt for this particle
    virtual real get_allowed_dt() const = 0;

    void set_compression(real compression) {
        dg_p = Matrix2(compression); // 1.0f = no compression
    }

    void calculate_kernels() {
        kernel_calc_counter++;
        Vector2 fpos = glm::fract(pos);
        real i_w[4], i_dw[4], j_w[4], j_dw[4];
        for (int i = -1; i < 3; i++) {
            i_w[i + 1] = w(fpos.x - i);
            i_dw[i + 1] = dw(fpos.x - i);
        }
        for (int j = -1; j < 3; j++) {
            j_w[j + 1] = w(fpos.y - j);
            j_dw[j + 1] = dw(fpos.y - j);
        }
        for (int i = -1; i < 3; i++) {
            for (int j = -1; j < 3; j++) {
                const Vector2 c_w = Vector2(i_w[i + 1], j_w[j + 1]);
                const Vector2 c_gw = Vector2(i_dw[i + 1], j_dw[j + 1]);
                const Vector2 t_gw = Vector2(c_gw.x * c_w.y, c_gw.y * c_w.x);
                const real t_w = c_w.x * c_w.y;
                cache_w[(i + 1) * 4 + j + 1] = t_w;
                cache_gw[(i + 1) * 4 + j + 1] = t_gw;
                cache_dg_gw[(i + 1) * 4 + j + 1] = glm::transpose(dg_e) * t_gw;
            }
        }
        mipos.x = int(floor(pos.x)) - 1;
        mipos.y = int(floor(pos.y)) - 1;
    }

    int get_cache_index(const Index2D &ind) const {
        return (ind.i - mipos.x) * 4 + (ind.j - mipos.y);
    }

    real get_cache_w(const Index2D &ind) const {
        return cache_w[get_cache_index(ind)];
    }

    Vector2 get_cache_gw(const Index2D &ind) const {
        return cache_gw[get_cache_index(ind)];
    }

    bool operator==(const MPMParticle &o) {
        return id == o.id;
    }

    virtual void calculate_force() {};

    virtual void plasticity() {};

    virtual void resolve_collision(const LevelSet2D &levelset) {
        real phi = levelset.sample(pos.x, pos.y);
        if (phi < 0) {
            pos -= levelset.get_normalized_gradient(pos) * phi;
        }
    }

    virtual void print() {
        P(pos);
        P(v);
        P(dg_e);
        P(dg_p);
    }
};


// Snow particle
struct EPParticle : MPMParticle {
    real theta_c, theta_s;
    real hardening;
    real mu_0, lambda_0;

    std::pair<real, real> get_lame_parameters() const {
        real j_e = det(dg_e);
        real j_p = det(dg_p);
        real e = std::exp(std::min(hardening * (1.0f - j_p), 5.0f));
        real mu = mu_0 * e;
        real lambda = lambda_0 * e;
        return {mu, lambda};
    }

    Matrix2 get_energy_gradient() {
        real j_e = det(dg_e);
        real j_p = det(dg_p);
        std::pair<real, real> lame_parameters;
        lame_parameters = get_lame_parameters();
        Matrix2 r, s;
        polar_decomp(dg_e, r, s);
        CV(r);
        CV(s);
        if (!is_normal(r) || !is_normal(s) || !is_normal(std::get<0>(lame_parameters)) || !is_normal(
                std::get<1>(lame_parameters))) {
            // debug code
            P(dg_e);
            P(dg_p);
            P(r);
            P(s);
            P(j_e);
            P(j_p);
        }
        return 2 * std::get<0>(lame_parameters) * (dg_e - r) +
               std::get<1>(lame_parameters) * (j_e - 1) * j_e * glm::inverse(glm::transpose(dg_e));
    }

    void plasticity() override {
        Matrix2 svd_u, sig, svd_v;
        svd(dg_e, svd_u, sig, svd_v);
        sig[0][0] = clamp(sig[0][0], 1.0f - theta_c, 1.0f + theta_s);
        sig[1][1] = clamp(sig[1][1], 1.0f - theta_c, 1.0f + theta_s);
        dg_e = svd_u * sig * glm::transpose(svd_v);
        dg_p = glm::inverse(dg_e) * dg_cache;
        svd(dg_p, svd_u, sig, svd_v);
        sig[0][0] = clamp(sig[0][0], 0.01f, 10.0f);
        sig[1][1] = clamp(sig[1][1], 0.01f, 10.0f);
        dg_p = svd_u * sig * glm::transpose(svd_v);
        //if (frobenius_norm(dg_p) > 50 || !is_normal(dg_p)) {
        //    P(dg_e);
        //    P(dg_p);
        //    P(dg_cache);
        //    P(svd_u);
        //    P(sig);
        //    P(svd_v);
        //    assert_info(is_normal(dg_p), "Abnormal dg_p");
        //}
    }

    void calculate_force() override {
        tmp_force = -vol * get_energy_gradient() * glm::transpose(dg_e);
        if (!is_normal(tmp_force)) {
            // debug code
            P(dg_e);
            P(get_energy_gradient());
        }
    };

    virtual real get_allowed_dt() const override {
        auto lame = get_lame_parameters();
        return 1e6f / std::max(std::get<0>(lame), std::get<1>(lame));
    }
};

// Sand particle
struct DPParticle : MPMParticle {
    real h_0, h_1, h_2, h_3;
    real lambda_0, mu_0;
    real alpha;
    real q = 0.0f;
    real phi_f;

    void project(Matrix2 sigma, real alpha, Matrix2 &sigma_out, real &out) {
        const real d = 2;
        Matrix2 epsilon(log(sigma[0][0]), 0.0f, 0.0f, log(sigma[1][1]));
        real tr = epsilon[0][0] + epsilon[1][1];
        Matrix2 epsilon_hat = epsilon - (tr) / d * Matrix2(1.0f);
        real epsilon_for = sqrt(epsilon[0][0] * epsilon[0][0] + epsilon[1][1] * epsilon[1][1]);
        real epsilon_hat_for = sqrt(epsilon_hat[0][0] * epsilon_hat[0][0] + epsilon_hat[1][1] * epsilon_hat[1][1]);
        if (epsilon_hat_for <= 0 || tr > 0.0f) {
            sigma_out = Matrix2(1.0f);
            out = epsilon_for;
        } else {
            real delta_gamma = epsilon_hat_for + (d * lambda_0 + 2 * mu_0) / (2 * mu_0) * tr * alpha;
            if (delta_gamma <= 0) {
                sigma_out = sigma;
                out = 0;
            } else {
                Matrix2 h = epsilon - delta_gamma / epsilon_hat_for * epsilon_hat;
                sigma_out = Matrix2(exp(h[0][0]), 0, 0, exp(h[1][1]));
                out = delta_gamma;
            }
        }
    }

    void calculate_force() override {
        Matrix2 u, v, sig, dg = dg_e;
        svd(dg_e, u, sig, v);

        Matrix2 log_sig(log(sig[0][0]), 0.0f, 0.0f, log(sig[1][1]));
        Matrix2 inv_sig(1.0f / (sig[0][0]), 0.0f, 0.0f, 1.0f / (sig[1][1]));
        Matrix2 center = 2 * mu_0 * inv_sig * log_sig + lambda_0 * (log_sig[0][0] + log_sig[1][1]) * inv_sig;

        tmp_force = -vol * (u * center * glm::transpose(v)) * glm::transpose(dg);
    }

    void plasticity() override {
        Matrix2 u, v, sig;
        svd(dg_e, u, sig, v);
        Matrix2 t = Matrix2(1.0);
        real delta_q = 0;
        project(sig, alpha, t, delta_q);
        Matrix2 rec = u * sig * glm::transpose(v);
        Matrix2 diff = rec - dg_e;
        if (!(frobenius_norm(diff) < 1e-4f)) {
            // debug code
            P(dg_e);
            P(rec);
            P(u);
            P(sig);
            P(v);
            error("SVD error\n");
        }
        dg_e = u * t * glm::transpose(v);
        dg_p = v * glm::inverse(t) * sig * glm::transpose(v) * dg_p;
        q += delta_q;
        real phi = h_0 + (h_1 * q - h_3) * expf(-h_2 * q);
        phi_f = phi;
        alpha = sqrtf(2.0f / 3.0f) * (2.0f * sin(phi * pi / 180.0f)) / (3.0f - sin(phi * pi / 180.0f));
    }

    real get_allowed_dt() const override {
        return 0.0f;
    }
};

TC_NAMESPACE_END

