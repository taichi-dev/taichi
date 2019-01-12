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
  Vector x(dim), v(dim);
  auto i = ind(), j = ind();
  auto l0 = var<float32>(), stiffness = var<float32>();
  auto neighbour = var<int32>();

  const auto h = 1.0e-2f;
  const auto viscous = 2.0e0f;
  const auto grav = -9.81;

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
      particle.place(x(i));
      particle.place(v(i));
    }
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        K_self(i, j) = var<float32>();
        particle.place(K_self(i, j));
      }
    }
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
  });

  return;
#if (0)
  auto build_stiffness = kernel(springs, [&] {
    auto u = i;
    auto v = springs[j];

    Matrix K(3, 3);

    atomic_add(fe(p(0)), fe0 * p(0).fixed);
    atomic_addd(fe(p(1)), -fe0 * p(1).fixed);

    Matrix UUt = outer_product(U, U);

    auto k = h * h * ((stiffness - f / l) * UUt + f / l * I);

    if
      !p(0).fixed land !p(1).fixed atomic_add(K(p(0), p(0)), k);
    atomic_add(K(p(0), p(1)), -k);
    atomic_add(K(p(1), p(0)), -k);
    atomic_add(K(p(1), p(1)), k);
    else if (p(0).fixed) K(p(0), p(0)) = k;
    if (p(1).fixed)
      K(p(1), p(1)) = k;
  });

  auto time_step = [&] {
    var r = f - (MDK * points.v);
    var p = r;
    var iter = 0;
    var normr2 = r'*r; for (int iter = 0; iter < 50; iter++) {
    /*
    Ap = MDK * p;
    denom = p'*Ap;
    alpha = normr2 / denom;
    points.v = points.v + alpha * p;
    normr2old = normr2;
    r = r - alpha * Ap;
    normr2 = r'*r;
    beta = normr2 / normr2old;
    p = r + beta * p;
    iter = iter + 1;
    */
  } points.x = points.x + h * points.v;
};
#endif
}

TLANG_NAMESPACE_END
