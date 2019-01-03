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
  auto hardening = 10.0_f, E = 1e4_f, nu = 0.2_f;
  // real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 *
  // nu));

  int dim = 2;

  Vector particle_x(dim), particle_v(dim);
  Matrix particle_F(dim, dim), particle_C(dim, dim);
  Real particle_J;

  Vector grid_v(dim);
  Real grid_m;

  Real Jp;

  int n_particles = 8192;
  Program prog(Arch::x86_64);

  prog.config.group_size = 1;
  prog.config.num_groups = 8;

  auto index = ind();
  auto grid_index = ind();

  layout([&]() {
    auto place = [&](Expr &expr) {
      expr = variable(DataType::f32);
      root.fixed(index, n_particles).place(expr);
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
    grid_m = variable(DataType::f32);

    root.fixed(grid_index, n * n).forked().place(grid_v(0), grid_v(1), grid_m);
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto clear_buffer = kernel(grid_m, [&]() {
    grid_v[grid_index](0) = imm(0.0_f);
    grid_v[grid_index](1) = imm(0.0_f);
    grid_m[grid_index] = imm(0.0_f);
  });

  auto p2g = kernel(particle_x(0), [&]() {
    auto x = particle_x[index];
    auto v = particle_v[index];
    // auto F = particle_F[index];
    auto C = particle_C[index];
    auto J = particle_J[index];

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

  auto grid_op = kernel(grid_m, [&]() {
    auto node = grid_index;
    auto v0 = load(grid_v[node](0));
    auto v1 = load(grid_v[node](1));
    auto m = load(grid_m[node]);

    // auto inv_m = imm(1.0_f) / max(m, imm(1e-37_f));
    auto inv_m = imm(1.0_f) / m;
    inv_m.name("inv_m");
    auto mask = cmp_lt(imm(0.0_f), m);
    mask.name("mask");
    v0 = select(mask, v0 * inv_m, imm(0.0_f)).name("v0");
    v1 = select(mask, v1 * inv_m + imm(dt * -200_f), imm(0.0_f)).name("v1");

    {
      auto i = node >> imm((int)bit::log2int(n));
      auto j = node & imm(n - 1);
      auto dist =
          min(min(i - imm(5), j - imm(5)), min(imm(n - 5) - i, imm(n - 5) - j));
      auto mask = cast<float32>(max(min(dist, imm(1)), imm(0)));
      v0 = v0 * mask;
      v1 = v1 * mask;
    }

    grid_v[node](0) = v0;
    grid_v[node](1) = v1;
  });

  auto g2p = kernel(particle_x(0), [&]() {
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

  int scale = 8;
  GUI gui("MPM", n * scale, n * scale);

  for (int i = 0; i < n_particles; i++) {
    particle_x(0).val<float32>(i) = 0.35_f + rand() * 0.3_f;
    particle_x(1).val<float32>(i) = 0.35_f + rand() * 0.3_f;
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
    gui.screenshot(fmt::format("images/{:04d}.png", f));
  }
};
TC_REGISTER_TASK(mpm);

auto advection = []() {
  bool use_adapter = false;

  const int dim = 2;

  const int n = 1024, nattr = 4;
  auto x = ind(), y = ind();

  Float attr[dim][nattr], v[dim];

  Program prog(Arch::x86_64);

  prog.config.group_size = use_adapter ? nattr : 1;
  prog.config.num_groups = use_adapter ? 8 : 8;

  layout([&]() {
    for (int k = 0; k < dim; k++) {
      if (use_adapter) {
        TC_NOT_IMPLEMENTED
        for (int i = 0; i < nattr; i++) {
          // prog.buffer(k).range(n * n).stream(0).group(0).place(attr[k][i]);
        }
        // prog.buffer(2).range(n * n).stream(0).group(0).place(v[k]);
      } else {
        for (int i = 0; i < nattr; i++) {
          attr[k][i] = var<float32>();
          root.fixed({x, y}, {n, n}).place(attr[k][i]);
        }
        v[k] = var<float32>();
        root.fixed({x, y}, {n, n}).place(v[k]);
      }
    }
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto clamp = [](const Expr &e) { return min(max(imm(0), e), imm(n - 2)); };

  auto func = kernel(attr[0][0], [&]() {
    // ** gs = 2

    auto vx = v[0][x, y];
    auto vy = v[1][x, y];

    auto offset_x = floor(vx);
    auto offset_y = floor(vy);
    auto wx = vx - offset_x;
    auto wy = vy - offset_y;

    if (use_adapter) {
      prog.adapter(0).set(2, 1).convert(offset_x, offset_y);
      prog.adapter(1).set(2, 1).convert(wx, wy);
    }

    // ** gs = 1
    auto new_x = cast<int32>(offset_x + cast<float32>(x));
    auto new_y = cast<int32>(offset_y + cast<float32>(y));


    new_x = clamp(new_x);
    new_y = clamp(new_y);

    // weights
    auto w00 = (imm(1.0f) - wx) * (imm(1.0f) - wy);
    auto w01 = (imm(1.0f) - wx) * wy;
    auto w10 = wx * (imm(1.0f) - wy);
    auto w11 = wx * wy;

    if (use_adapter) {
      prog.adapter(2).set(1, 4).convert(w00, w01, w10, w11);
      // prog.adapter(3).set(1, 4).convert(node);
    }

    // ** gs = 4
    for (int k = 0; k < nattr; k++) {
      auto v00 = attr[0][k][new_x, new_y].name("v00");
      auto v01 = attr[0][k][new_x, new_y + imm(1)].name("v01");
      auto v10 = attr[0][k][new_x + imm(1), new_y].name("v10");
      auto v11 = attr[0][k][new_x + imm(1), new_y + imm(1)].name("v11");
      attr[1][k][x, y] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
      attr[1][k][x, y].name(fmt::format("output{}", k));
    }
  });

  auto swap_buffers = kernel(attr[0][0], [&] {
    for (int i = 0; i < nattr; i++) {
      attr[0][i][x, y] = attr[1][i][x, y];
    }
  });

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < nattr; k++) {
        attr[0][k].val<float32>(i, j) = i % 128 / 128.0_f;
      }
      real s = 20.0_f / n;
      v[0].val<float32>(i, j) = s * (j - n / 2);
      v[1].val<float32>(i, j) = -s * (i - n / 2);
    }
  }

  GUI gui("Advection", n, n);

  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 10; t++) {
      TC_TIME(func());
      TC_TIME(swap_buffers());
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < nattr; k++) {
          gui.buffer[i][j] = Vector4(attr[1][k].val<float32>(i, j));
        }
      }
    }
    gui.update();
    // gui.screenshot(fmt::format("images/{:04d}.png", f));
  }
};
TC_REGISTER_TASK(advection);

auto test_snode = [&]() {
  Program prog(Arch::x86_64);

  auto i = Expr::index(0);
  auto u = variable(DataType::i32);

  int n = 128;

  // All data structure originates from a "root", which is a forked node.
  prog.layout([&] { root.fixed(i, n).place(u); });

  for (int i = 0; i < n; i++) {
    u.val<int32>(i) = i + 1;
  }

  for (int i = 0; i < n; i++) {
    TC_ASSERT(u.val<int32>(i) == i + 1);
  }
};

TC_REGISTER_TASK(test_snode);

TC_NAMESPACE_END
