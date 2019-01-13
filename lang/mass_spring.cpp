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
  Expr t = vec(0) * vec(0);
  for (int i = 1; i < (int)vec.entries.size(); i++) {
    t = t + vec.entries[i] * vec.entries[i];
  }
  return sqrt(t);
}

TC_TEST("mass_spring") {
  Program prog;
  prog.config.simd_width = 1;
  // TC_WARN("optimization off");
  // prog.config.external_optimization_level = 1;

  const int dim = 3;
  auto i = ind(), j = ind();

  Matrix K(dim, dim), K_self(dim, dim);
  Vector x(dim), v(dim), fmg(dim), p(dim), r(dim), Ap(dim), vec(dim);

  auto mass = var<float32>(), fixed = var<float32>();
  auto l0 = var<float32>(), stiffness = var<float32>();
  auto neighbour = var<int32>();
  auto alpha = var<float32>(), beta = var<float32>();
  auto denorm = var<float32>(), normr2 = var<float32>();

  const auto h = 1.0e-2_f;
  const auto viscous = 2_f;
  const auto grav = -9.81_f;
  // const auto grav = 0;

  int n, m;
  std::FILE *f = std::fopen("data/bunny.txt", "r");
  TC_ASSERT(f);
  fscanf(f, "%d%d", &n, &m);
  TC_P(n);
  int max_n = bit::least_pot_bound(n);
  TC_P(m);
  int max_edges = 64;
  TC_P(max_n);

  layout([&] {
    root.fixed(i, max_n)//  .multi_threaded()
        .dynamic(j, max_edges)
        .place(neighbour);
    auto &fork2 = root.fixed(i, max_n);

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        K(i, j) = var<float32>();
      }
    }
    auto place_fixed = [&](Expr expr) {
      fork2.fixed(j, max_edges).place(expr);
    };

    auto &fork3 = root.fixed(i, max_n);//.fixed(j, max_edges / 8);
    auto place_blocked = [&](Expr expr) { fork3.fixed(j, max_edges).place(expr); };

    place_fixed(l0);
    place_fixed(stiffness);
    for (auto &e : K.entries) {
      place_blocked(e);
    }

    auto &particle = root.fixed(i, max_n);
    for (int i = 0; i < dim; i++) {
      x(i) = var<float32>();
      v(i) = var<float32>();
      fmg(i) = var<float32>();
      r(i) = var<float32>();
      p(i) = var<float32>();
      Ap(i) = var<float32>();
      vec(i) = var<float32>();
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

    for (int d = 0; d < dim; d++) {
      root.fixed(i, max_n).place(vec(d));
    }
    root.place(alpha);
    root.place(beta);
    root.place(denorm);
    root.place(normr2);
  });

  auto clear_matrix = kernel(neighbour, [&] {
    for (int t = 0; t < dim; t++) {
      for (int d = 0; d < dim; d++) {
        K[i, j](t, d) = 0;
      }
    }
  });

  auto build_matrix = kernel(neighbour, [&] {
    auto k = neighbour[i, j];
    auto dx = x[k] - x[i];
    auto l = length(dx) +
             imm(0.00000000001_f);  // NOTE: this may lead to a difference
    auto U = dx * (imm(1.0_f) / l);
    auto s = load(stiffness[i, j]);
    auto f = s * (l - l0[i, j]);
    auto fe0 = (imm(1.0_f) - fixed[i]) * imm(h) * f * U;

    Matrix UUt = outer_product(U, U);

    auto k_e = imm(h * h) * ((s - f * (imm(1.0_f) / l)) * UUt);
    for (int t = 0; t < dim; t++)
      k_e(t, t) = k_e(t, t) + f / l * imm(h * h);

    k_e = k_e * (imm(1.0_f) - fixed[i]);

    K[i, j] = -k_e;

    for (int d = 0; d < dim; d++) {
      reduce(fmg[i](d), fe0(d));
      for (int t = 0; t < dim; t++) {
        reduce(K_self(d, t)[i], k_e(d, t));
      }
    }
  });

  auto preprocess_particles = kernel(x(0), [&] {
    auto fmg_t = mass[i] * v[i];
    fmg_t(1) = fmg_t(1) +
               imm(h * grav) * mass[i] * (imm(1.0_f) - fixed[i]);  // gravity
    fmg_t = fmg_t * (imm(1.0_f) - fixed[i]);
    fmg[i] = fmg_t;
    for (int t = 0; t < dim; t++) {
      K_self[i](t, t) = select(cast<int>(load(fixed[i])), imm(1.0_f),
                               mass[i] + imm(h * viscous));
      for (int d = 0; d < dim; d++) {
        if (t != d)
          K_self[i](t, d) = imm(0.0_f);
      }
    }

  });

  auto advect = kernel(mass, [&] { x[i] = x[i] + imm(h) * v[i]; });

  auto copy_r_to_p = kernel(mass, [&] { p[i] = r[i]; });

  auto compute_Ap1 = kernel(neighbour, [&] {
    kernel_name("compute_Ap1");
    Vector ve(3);
    auto offset = neighbour[i, j];
    for (int d = 0; d < dim; d++) {
      ve(d) = load(vec(d)[offset]);
    }
    auto tmp = K[i, j] * vec[offset];
    for (int d = 0; d < dim; d++) {
      reduce(Ap[i](d), tmp(d));
    }
  });

  auto compute_Ap2 = kernel(mass, [&] { Ap[i] = K_self[i] * vec[i]; });

  auto compute_Ap = [&] {
    compute_Ap2();
    TC_TIME(compute_Ap1());
  };

  auto compute_denorm = kernel(mass, [&] {
    reduce(global(denorm), p[i].element_wise_prod(Ap[i]).sum());
  });

  auto compute_r = kernel(mass, [&] { r[i] = fmg[i] - Ap[i]; });

  auto compute_v = kernel(mass, [&] { v[i] = v[i] + global(alpha) * p[i]; });

  auto compute_normr2 = kernel(mass, [&] {
    reduce(global(normr2), r[i].element_wise_prod(r[i]).sum());
  });

  auto compute_r2 = kernel(mass, [&] { r[i] = r[i] - global(alpha) * Ap[i]; });

  auto compute_p = kernel(mass, [&] { p[i] = r[i] + global(beta) * p[i]; });

  auto copy_v_to_vec = kernel(mass, [&] { vec[i] = v[i]; });

  auto copy_p_to_vec = kernel(mass, [&] { vec[i] = p[i]; });

  auto print_vector = [&](std::string name, Vector e) {
    return;
    fmt::print(name);
    for (int i = 0; i < n; i++) {
      fmt::print(" {} {} {}, ", e(0).val<float>(i), e(1).val<float>(i),
                 e(2).val<float>(i));
    }
    fmt::print("\n");
  };

  std::vector<int> degrees(n, 0);
  auto time_step = [&] {
    clear_matrix();
    preprocess_particles();
    TC_TIME(build_matrix());

    // print_vector("x", x);
    // print_vector("v", v);

    /*
    for (int u = 0; u < n; u++) {
      printf("u: %d\n", u);
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
          printf("K_self %f\n", K_self(i, j).val<float32>(u));
        }
      }
      for (int k = 0; k < degrees[u]; k++) {
        int v = neighbour.val<int32>(u, k);
        printf("v: %d\n", v);
        for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
            printf("K %f\n", K(i, j).val<float32>(u, k));
          }
        }
      }

    }
    */

    // print_vector("fmg", fmg);
    copy_v_to_vec();  // vec = v;
    // print_vector("vec", vec);
    compute_Ap();  // Ap = K vec
    // print_vector("Ap", Ap);
    compute_r();  // r = fmg - Ap

    print_vector("r", r);
    copy_r_to_p();  // p = r
    // print_vector("p", p);
    normr2.val<float32>() = 0;
    compute_normr2();  // normr2 = r' * r
    // TC_P(normr2.val<float32>());
    auto h_normr2 = normr2.val<float32>();
    int cnt = 0;
    for (int i = 0; i < 50; i++) {
      // TC_P(i);
      cnt = i;
      if (h_normr2 < 1e-8f) {
        break;
      }
      copy_p_to_vec();        // vec = p
      while (1)compute_Ap();  // Ap = K vec
      // print_vector("Ap", Ap);
      denorm.val<float32>() = 0;
      compute_denorm();  // denorm = p' Ap
      alpha.val<float32>() =
          normr2.val<float32>() / (denorm.val<float32>() + 1e-30f);
      // TC_P(alpha.val<float32>());
      // print_vector("p", p);
      compute_v();   // v = v + alpha * p
      compute_r2();  // r = r - alpha * Ap
      // print_vector("r", r);
      // normr2.val()
      auto normr2_old = normr2.val<float32>();
      // TC_P(normr2_old);
      normr2.val<float32>() = 0;
      compute_normr2();  // normr2 = r'r
      // TC_P(normr2.val<float32>());
      if (normr2.val<float32>() < 1e-8_f)
        break;
      beta.val<float32>() = normr2.val<float32>() / normr2_old;
      compute_p();  // p = r + beta * p
    }
    TC_INFO("# cg iterations {}", cnt);
    print_vector("x", x);
    print_vector("v", v);
    print_vector("r", r);
    advect();
  };

  for (int i = 0; i < n; i++) {
    float32 x_, y_, z_, fixed_;
    fscanf(f, "%f%f%f%f", &x_, &y_, &z_, &fixed_);
    x(0).val<float32>(i) = x_;
    x(1).val<float32>(i) = y_;
    x(2).val<float32>(i) = z_;
    fixed.val<float32>(i) = fixed_;
  }

  for (int i = 0; i < m; i++) {
    int u, v;
    float32 l0_, stiffness_;
    fscanf(f, "%d%d%f%f", &u, &v, &l0_, &stiffness_);

    if (std::max(degrees[v], degrees[u]) >= max_edges) {
      TC_WARN("Skipping edge");
      continue;
    }

    l0.val<float>(u, degrees[u]) = l0_;
    l0.val<float>(v, degrees[v]) = l0_;
    stiffness.val<float>(u, degrees[u]) = stiffness_;
    stiffness.val<float>(v, degrees[v]) = stiffness_;

    neighbour.val<int>(u, degrees[u]) = v;
    neighbour.val<int>(v, degrees[v]) = u;

    degrees[u]++;
    degrees[v]++;
  }

  int max_degrees = 0;
  int total_degrees = 0;
  for (int i = 0; i < n; i++) {
    float32 mass_;
    fscanf(f, "%f", &mass_);
    mass.val<float32>(i) = mass_;
    max_degrees = std::max(max_degrees, degrees[i]);
    total_degrees += degrees[i];
  }
  TC_P(max_degrees);
  TC_P(1.0_f * total_degrees / n);

  for (int i = 0; i < 500; i++) {
    for (int i = 0; i < 10; i++)
      TC_TIME(time_step());
    std::vector<Vector3> parts;
    for (int i = 0; i < n; i++) {
      parts.push_back(Vector3(x(0).val<float32>(i), x(1).val<float32>(i),
                              x(2).val<float32>(i)));
    }
    write_partio(parts, fmt::format("ms_output/{:05d}.bgeo", i));
  }
}

TLANG_NAMESPACE_END
