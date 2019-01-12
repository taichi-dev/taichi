#include <iostream>
#include "util.h"
#include "tlang.h"
#include <taichi/common/util.h>
#include <taichi/math.h>
#include <taichi/common/testing.h>
#include <taichi/system/timer.h>
#include <unordered_map>

using namespace taichi;
using Tlang::measure_cpe;

TLANG_NAMESPACE_BEGIN

Expr length(Vector vec) {
  TC_WARN("not implemented");
  return vec(0);
}

TC_TEST("mass_spring") {
  Program prog;

  const int dim = 3;
  Vector x(dim), v(dim), fe(dim), fmg(dim), p(dim), r(dim), Ap(dim);
  auto mass = var<float32>(), fixed = var<float32>();
  auto i = ind(), j = ind();
  auto l0 = var<float32>(), stiffness = var<float32>();
  auto neighbour = var<int32>();
  auto alpha = var<int32>(), beta = var<int32>();
  auto denorm = var<float32>(), normr2 = var<float32>();

  const auto h = 1.0e-2_f;
  const auto viscous = 2_f;
  const auto grav = -9.81_f;

  int n, m;
  Matrix K(dim, dim), K_self(dim, dim);
  std::FILE *f = std::fopen("data/bunny.txt", "r");
  TC_ASSERT(f);
  fscanf(f, "%d%d", &n, &m);
  TC_P(n);
  TC_P(m);
  std::vector<int> degrees(n, 0);
  for (int i = 0; i < n; i++) {
    float32 x, y, z, fixed;
    fscanf(f, "%f%f%f%f", &x, &y, &z, &fixed);
  }

  for (int i = 0; i < m; i++) {
    int u, v, l0, stiffness;
    fscanf(f, "%d%d%f%f", &u, &v, &l0, &stiffness);
    degrees[u]++;
    degrees[v]++;
  }

  int max_degrees = 0;
  int total_degrees = 0;
  for (int i = 0; i < n; i++) {
    float32 mass;
    fscanf(f, "%f", &mass);
    max_degrees = std::max(max_degrees, degrees[i]);
    total_degrees += degrees[i];
  }
  TC_P(max_degrees);
  TC_P(1.0_f * total_degrees / n);

  layout([&] {
    auto &fork = root.fixed(i, 65536).dynamic(j, 64);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        K(i, j) = var<float32>();
        fork.place(K(i, j));
      }
    }
    fork.place(l0, stiffness, neighbour);

    auto &particle = root.fixed(i, 65536);
    for (int i = 0; i < dim; i++) {
      x(i) = var<float32>();
      v(i) = var<float32>();
      fmg(i) = var<float32>();
      r(i) = var<float32>();
      p(i) = var<float32>();
      Ap(i) = var<float32>();
      particle.place(x(i));
      particle.place(v(i));
      particle.place(fmg(i));
      particle.place(r(i));
      particle.place(p(i));
      particle.place(Ap(i));
    }
    particle.place(fixed);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        K_self(i, j) = var<float32>();
        particle.place(K_self(i, j));
      }
    }
    particle.place(mass);

    root.place(alpha);
    root.place(beta);
    root.place(denorm);
    root.place(normr2);
  });

  auto build_matrix = kernel(neighbour, [&] {
    auto k = neighbour[i, j];
    auto dx = x[i] - x[k];
    auto l = length(dx);
    auto U = dx * (imm(1.0_f) / l);
    auto s = stiffness[i, j];
    auto f = s * (l - l0[i, j]);
    auto fe0 = imm(h) * f * U;

    Matrix UUt = outer_product(U, U);

    auto k_e = imm(h * h) * (s - f * (imm(1.0_f) / l) * UUt);
    for (int t = 0; t < dim; t++)
      k_e(t, t) = k_e(t, t) + f / l;

    K_self[i] = K_self[i] + k_e;
    K[i, j] = K[i, j] - k_e;

    fe[i] = fe[i] + fe0;
  });

  auto preprocess_particles = kernel(x(0), [&] {
    fmg[i] = mass[i] * v[i];
    fmg[i](1) = fmg[i](1) + imm(h * grav) * mass[i];  // gravity
    for (int t = 0; t < dim; t++) {
      K_self(t, t) = K_self(t, t) + select(cast<int>(fixed[i]), imm(1.0_f),
                                           mass[i] + imm(h * viscous));
    }
  });

  auto advect = kernel(mass, [&] { x[i] = x[i] + imm(h) * v[i]; });

  auto copy_r_to_p = kernel(mass, [&] { p[i] = r[i]; });

  auto compute_Ap1 = kernel(K(0, 0), [&] {
    auto tmp = K[i, j] * p[neighbour[i, j]];
    for (int d = 0; d < dim; d++)
      reduce(Ap[i](d), tmp[i](d));
  });

  auto compute_Ap2 = kernel(K(0, 0), [&] { Ap[i] = Ap[i] + K_self[i] * p[i]; });

  auto compute_Ap = [&] {
    compute_Ap1();
    compute_Ap2();
  };

  auto compute_denorm = kernel(mass, [&] {});

  auto compute_r = kernel(mass, [&] { r[i] = fe[i] - Ap[i]; });

  auto compute_v = kernel(mass, [&] { v[i] = v[i] + alpha * p[i]; });

  auto compute_r2 = kernel(mass, [&] {
    // reduce(normr2, );
  });

  auto compute_normr2 = kernel(mass, [&] { r[i] = r[i] - alpha * Ap[i]; });

  auto compute_p = kernel(mass, [&] { p[i] = r[i] + beta * p[i]; });

  auto time_step = [&] {
    compute_normr2();
    copy_r_to_p();
    auto h_normr2 = normr2.val<float32>();
    for (int i = 0; i < 50; i++) {
      compute_Ap();
      compute_denorm();
      alpha.val<float32>() = normr2.val<float32>() / denorm.val<float32>();
      compute_v();
      compute_r2();
      compute_normr2();
      auto normr2_old = normr2.val<float32>();
      TC_P(normr2.val<float32>());
      beta.val<float32>() = normr2.val<float32>() / normr2_old;
      compute_p();
    }
    advect();
  };

  for (int i = 0; i < 100; i++) {
    time_step();
  }
}

TLANG_NAMESPACE_END
