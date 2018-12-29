#include "tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

Matrix outer_product(Vector a, Vector b) {
  TC_ASSERT(a.m == 1);
  TC_ASSERT(b.m == 1);
  Matrix m(a.n, b.n);
  for (int i = 0; i < a.n; i++) {
    for (int j = 0; j < b.n; j++) {
      m(i, j) = a(i) * b(j);
    }
  }
  return m;
}

auto mpm = []() {
  bool use_adapter = true;

  constexpr int n = 128;  // grid_resolution
  const real dt = 3e-5_f, frame_dt = 1e-3_f, dx = 1.0_f / n,
             inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto hardening = 10.0_f, E = 1e3_f, nu = 0.2_f;
  real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

  int dim = 2;

  Vector particle_x(dim), particle_v(dim);
  Matrix particle_F(dim, dim), particle_C(dim, dim);
  Real particle_J;

  Vector grid_v(dim);
  Real grid_m;

  Real Jp;

  int n_particles = 4000;
  Program prog(Arch::x86_64);
  prog.general_scatter = true;

  prog.config.group_size = 1;
  prog.config.num_groups = 8;

  prog.layout([&]() {
    int counter = 0;
    auto place = [&](Expr &expr) {
      prog.buffer(0).range(n_particles).stream(counter++).group(0).place(expr);
    };
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_C(i, j));
      }
      place(particle_x(i));
      place(particle_v(i));
    }
    place(particle_J);

    prog.buffer(1).range(n * n).stream().group().place(grid_v(0), grid_v(1),
                                                       grid_m);
    /*
    for (int k = 0; k < 2; k++) {
      prog.buffer(k).range(n * n).stream(i).group(0).place(attr[k][i]);
    }
    prog.buffer(2).range(n * n).stream(k).group(0).place(v[k]);
    */
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto p2g = prog.def([&]() {
    auto index = Expr::index(0);
    for_loop(index, {0, n_particles}, [&] {
      auto x = particle_x[index];
      auto v = particle_v[index];
      // auto F = particle_F[index];
      auto C = particle_C[index];
      auto J = particle_J[index];

      // ** gs = 2

      auto base_coord = floor(imm(inv_dx) * x - imm(0.5_f));
      auto fx = x * imm(inv_dx) - base_coord;

      Vector w[3];
      w[0] = imm(0.5_f) * sqr(imm(1.5_f) - fx);
      w[1] = imm(0.75_f) - sqr(fx - imm(1.0_f));
      w[2] = imm(0.5_f) * sqr(fx - imm(0.5_f));

      auto cauchy = imm(E) * (J - imm(1.0_f));
      auto affine = imm(particle_mass) * C;
      for (int i = 0; i < dim; i++) {
        affine(i, i) =
            affine(i, i) + imm(-4 * inv_dx * inv_dx * dt * vol) * cauchy;
      }

      // auto J = F(0, 0) * F(1, 1) - F(1, 0) * F(0, 1);
      auto base_offset =
          cast<int>(base_coord(0)) * imm(n) + cast<int>(base_coord(1));

      // scatter
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          auto dpos = Vector(dim);
          dpos(0) = imm(dx) * (cast<float32>(imm(i)) - fx(0));
          dpos(1) = imm(dx) * (cast<float32>(imm(j)) - fx(1));
          auto weight = w[i](0) * w[j](1);
          auto node = base_offset + imm(i * n + j);
          grid_v[node] =
              grid_v[node] + weight * (imm(particle_mass) * v + affine * dpos);
          grid_m[node] = grid_m[node] + imm(particle_mass) * weight;
        }
      }

    });
  });

  p2g();

  auto grid_op = prog.def([&]() {
    auto node = Expr::index(0);
    for_loop(node, {0, n * n}, [&] {
      auto v0 = load(grid_v[node](0));
      auto v1 = load(grid_v[node](1));
      auto m = load(grid_m[node]);

      // auto inv_m = imm(1.0_f) / max(m, imm(1e-37_f));
      auto inv_m = imm(1.0_f) / m;
      auto mask = cmp_ne(m, imm(0.0_f));
      v0 = select(mask, v0 * inv_m, imm(0.0_f));
      v1 = select(mask, v1 * inv_m + imm(dt * -200_f), imm(0.0_f));

      /*
      auto i = node >> imm((int)bit::log2int(n));
      auto j = node & imm(n - 1);
      auto dist =
          min(min(i - imm(5), j - imm(5)), min(imm(n - 5) - i, imm(n - 5) - j));
      auto mask = cast<float32>(max(min(dist, imm(1)), imm(0)));
      v0 = v0 * mask;
      v1 = v1 * mask;
      */

      grid_v[node](0) = v0;
      grid_v[node](1) = v1;
    });
  });

  auto g2p = prog.def([&]() {
    auto index = Expr::index(0);
    for_loop(index, {0, n_particles}, [&]() {
      auto x = particle_x[index];
      auto v = Vector(dim);
      // auto F = particle_F[index];
      auto C = Matrix(dim, dim);
      auto J = particle_J[index];

      for (int i = 0; i < dim; i++) {
        v(i) = imm(0.0_f);
        for (int j = 0; j < dim; j++) {
          C(i, j) = imm(0.0_f);
        }
      }

      auto base_coord = floor(imm(inv_dx) * x - imm(0.5_f));
      auto fx = x * imm(inv_dx) - base_coord;

      Vector w[3];
      w[0] = imm(0.5_f) * sqr(imm(1.5_f) - fx);
      w[1] = imm(0.75_f) - sqr(fx - imm(1.0_f));
      w[2] = imm(0.5_f) * sqr(fx - imm(0.5_f));

      // auto J = F(0, 0) * F(1, 1) - F(1, 0) * F(0, 1);
      auto base_offset =
          cast<int>(base_coord(0)) * imm(n) + cast<int>(base_coord(1));

      // scatter
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          auto dpos = Vector(dim);
          dpos(0) = cast<float32>(imm(i)) - fx(0);
          dpos(1) = cast<float32>(imm(j)) - fx(1);
          auto weight = w[i](0) * w[j](1);
          auto node = base_offset + imm(i * n + j);
          v = v + weight * grid_v[node];
          C = C + imm(4 * inv_dx) * outer_product(weight * grid_v[node], dpos);
        }
      }

      J = J * (imm(1.0_f) + imm(dt) * (C(0, 0) + C(1, 1)));

      particle_C[index] = C;
      particle_v[index] = v;
      particle_J[index] = J;
      x = x + imm(dt) * v;
      particle_x[index] = x;
    });
  });

  int scale = 8;
  GUI gui("MPM", n * scale, n * scale);

  for (int i = 0; i < n_particles; i++) {
    prog.data(particle_x(0), i) = 0.35_f + rand() * 0.3_f;
    prog.data(particle_x(1), i) = 0.35_f + rand() * 0.3_f;
    prog.data(particle_v(1), i) = -0.3_f;
    prog.data(particle_J, i) = 1_f;
  }

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 100; t++) {
      prog.clear_buffer(1);
      TC_TIME(p2g());
      TC_TIME(grid_op());
      TC_TIME(g2p());
    }
    for (int i = 0; i < n * scale; i++) {
      for (int j = 0; j < n * scale; j++) {
        gui.buffer[i][j].x =
            prog.data(grid_v(0), i / scale * n + j / scale) + 0.5;
        gui.buffer[i][j].y =
            prog.data(grid_v(1), i / scale * n + j / scale) + 0.5;
        gui.buffer[i][j].z =
            prog.data(grid_m, i / scale * n + j / scale) / particle_mass * 0.0 +
            0.5;
      }
    }

    gui.update();
    // gui.screenshot(fmt::format("images/{:04d}.png", f));
  }
};
TC_REGISTER_TASK(mpm);

TC_NAMESPACE_END

/*
TODO: arbitrary for loop (bounds using arbitrary constants)
 */
