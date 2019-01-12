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

TC_TEST("mass_spring") {
  Program prog;

  int n, m;
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
  for (int i = 0; i < n; i++) {
    float32 mass;
    fscanf(f, "%f", &mass);
    max_degrees = std::max(max_degrees, degrees[i]);
  }
  TC_P(max_degrees);
  return;

#if(0)
  auto K = var<float32>(), x = var<float32>();
  auto i = ind();
  auto l0 = var<float32>(), stiffness = var<float32>();
  layout([&] {
    root.hashed(i, 1024).fixed(i, 1024).pointer()
        .fixed(i, 256).place(x, y);
  });

  auto build_stiffness = kernel(springs, [&]{
    auto u = i;
    auto v = springs[j];

    Matrix K(3, 3);
    auto dx = x[u] - x[v];
    auto l = sqrt(norm(dx));
    auto U = dx * inv(l);

    auto f = stiffness*(l-l0);
    auto fe0 = h*f*U;

    atomic_add(fe(p(0)), fe0 * p(0).fixed);
    atomic_addd(fe(p(1)), -fe0 * p(1).fixed);

    Matrix UUt = outer_product(U, U);

    auto k = h*h*((stiffness-f/l)*UUt + f/l*I);

    if !p(0).fixed land !p(1).fixed
      atomic_add(K(p(0),p(0)), k);
      atomic_add(K(p(0),p(1)), -k);
      atomic_add(K(p(1),p(0)), -k);
      atomic_add(K(p(1),p(1)), k);
    else
      if (p(0).fixed)
        K(p(0),p(0)) = k;
      if (p(1).fixed)
        K(p(1),p(1)) = k;
  });


  auto time_step = [&] {
    var r = f - (MDK*points.v);
    var p = r;
    var iter = 0;
    var normr2 = r'*r;
    for (int iter = 0; iter < 50; iter++) {
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
    }
    points.x = points.x + h * points.v;

  };

  TC_TIME(stencil());
#endif
}

TLANG_NAMESPACE_END
