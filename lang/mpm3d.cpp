#include "tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

auto mpm3d = []() {
  Program prog(Arch::x86_64);
  prog.config.gcc_version = 7;
  bool use_adapter = true;

  constexpr int n = 64;  // grid_resolution
  const real dt = 3e-5_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto E = 1e4_f;
  // real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 *
  // nu));

  int dim = 3;

  Vector particle_x(dim), particle_v(dim);
  Matrix particle_F(dim, dim), particle_C(dim, dim);
  Real particle_J;

  Vector grid_v(dim);
  Real grid_m;

  Real Jp;

  int n_particles = 8192 / 4;

  auto p = ind();
  auto i = ind(), j = ind(), k = ind();

  layout([&]() {
    auto place = [&](Expr &expr) {
      expr = variable(DataType::f32);
      root.fixed(p, n_particles).place(expr);
    };
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_C(i, j));
      }
      place(particle_x(i));
      place(particle_v(i));
    }
    place(particle_J);

    grid_v(0) = variable(DataType::f32);
    grid_v(1) = variable(DataType::f32);
    grid_v(2) = variable(DataType::f32);
    grid_m = variable(DataType::f32);

    root.fixed({i, j, k}, {n, n, n})
        .forked()
        .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto clear_buffer = kernel(grid_m, [&]() {
    grid_v(0)[i, j, k] = imm(0.0_f);
    grid_v(1)[i, j, k] = imm(0.0_f);
    grid_m[i, j, k] = imm(0.0_f);
  });

  auto p2g = kernel(particle_x(0), [&]() {
    auto x = particle_x[p];
    auto v = particle_v[p];
    // auto F = particle_F[p];
    auto C = particle_C[p];
    auto J = particle_J[p];

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

    // scatter
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          auto dpos = Vector(dim);
          dpos(0) = imm(dx) * (imm(i * 1.0_f) - fx(0));
          dpos(1) = imm(dx) * (imm(j * 1.0_f) - fx(1));
          dpos(2) = imm(dx) * (imm(k * 1.0_f) - fx(2));
          auto weight = w[i](0) * w[j](1) * w[k](2);
          auto node = (cast<int32>(base_coord(0)) + imm(i),
                       cast<int32>(base_coord(1)) + imm(j),
                       cast<int32>(base_coord(2)) + imm(k));
          grid_v[node] =
              grid_v[node] + weight * (imm(particle_mass) * v + affine * dpos);
          grid_m[node] = grid_m[node] + weight * imm(particle_mass);
        }
      }
    }
  });

  auto grid_op = kernel(grid_m, [&]() {
    auto v0 = load(grid_v[i, j, k](0));
    auto v1 = load(grid_v[i, j, k](1));
    auto v2 = load(grid_v[i, j, k](2));
    auto m = load(grid_m[i, j, k]);

    // auto inv_m = imm(1.0_f) / max(m, imm(1e-37_f));
    auto inv_m = imm(1.0_f) / m;
    inv_m.name("inv_m");
    auto mask = cmp_lt(imm(0.0_f), m);
    mask.name("mask");
    v0 = select(mask, v0 * inv_m, imm(0.0_f)).name("v0");
    v1 = select(mask, v1 * inv_m + imm(dt * -200_f), imm(0.0_f)).name("v1");
    v2 = select(mask, v2 * inv_m, imm(0.0_f)).name("v2");

    {
      auto dist =
          min(min(i - imm(5), j - imm(5)), min(min(imm(n - 5) - i, imm(n - 5) - j),
              min(imm(n - 5) - k, imm(n - 5) - k)));
      auto mask = cast<float32>(max(min(dist, imm(1)), imm(0)));
      v0 = v0 * mask;
      v1 = v1 * mask;
      v2 = v2 * mask;
    }

    grid_v[i, j](0) = v0;
    grid_v[i, j](1) = v1;
    grid_v[i, j](2) = v2;
  });

  auto g2p = kernel(particle_x(0), [&]() {
    auto x = particle_x[p];
    auto v = Vector(dim);
    // auto F = particle_F[p];
    auto C = Matrix(dim, dim);
    auto J = particle_J[p];

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

    // scatter
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          auto dpos = Vector(dim);
          dpos(0) = imm(i * 1.0_f) - fx(0);
          dpos(1) = imm(j * 1.0_f) - fx(1);
          dpos(2) = imm(k * 1.0_f) - fx(1);
          auto weight = w[i](0) * w[j](1) * w[k](2);
          auto wv = weight * grid_v[cast<int32>(base_coord(0)) + imm(i),
                                    cast<int32>(base_coord(1)) + imm(j),
                                    cast<int32>(base_coord(2)) + imm(j)];
          v = v + wv;
          C = C + imm(4 * inv_dx) * outer_product(wv, dpos);
        }
      }
    }

    J = J * (imm(1.0_f) + imm(dt) * (C(0, 0) + C(1, 1) + C(2, 2)));
    x = x + imm(dt) * v;

    particle_C[p] = C;
    particle_v[p] = v;
    particle_J[p] = J;
    particle_x[p] = x;
  });

  int scale = 8;
  GUI gui("MPM", n * scale, n * scale);

  for (int i = 0; i < n_particles; i++) {
    particle_x(0).val<float32>(i) = 0.35_f + rand() * 0.3_f;
    particle_x(1).val<float32>(i) = 0.35_f + rand() * 0.3_f;
    particle_x(2).val<float32>(i) = 0.35_f + rand() * 0.3_f;
    particle_v(1).val<float32>(i) = -0.3_f;
    particle_J.val<float32>(i) = 1_f;
  }

  auto &canvas = gui.get_canvas();

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 200; t++) {
      TC_TIME(clear_buffer());
      TC_TIME(p2g());
      TC_TIME(grid_op());
      TC_TIME(g2p());
    }
    /*
    for (int i = 0; i < n * scale; i++) {
      for (int j = 0; j < n * scale; j++) {
        gui.buffer[i][j].x =
            grid_v(0).set<float32>(i / scale * n + j / scale) * 0.01 + 0.5;
        gui.buffer[i][j].y =
            prog.data(grid_v(1), i / scale * n + j / scale) * 0.01 + 0.5;
        gui.buffer[i][j].z = 1.0;
      }
    }
    */
    canvas.clear(0x112F41);
    for (int i = 0; i < n_particles; i++) {
      canvas
          .circle(particle_x(0).val<float32>(i), particle_x(1).val<float32>(i))
          .radius(2)
          .color(0x068587);
    }

    gui.update();
    // gui.screenshot(fmt::format("images/{:04d}.png", f));
  }
};
TC_REGISTER_TASK(mpm3d);

TC_NAMESPACE_END
