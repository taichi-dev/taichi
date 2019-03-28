#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <Partio.h>

TC_NAMESPACE_BEGIN

using namespace Tlang;

void write_partio(std::vector<Vector3> positions,
                  const std::string &file_name) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  Partio::ParticleAttribute posH;
  posH = parts->addAttribute("position", Partio::VECTOR, 3);
  for (auto p : positions) {
    int idx = parts->addParticle();
    float32 *p_p = parts->dataWrite<float32>(posH, idx);
    for (int k = 0; k < 3; k++)
      p_p[k] = 0.f;
    for (int k = 0; k < 3; k++)
      p_p[k] = p[k];
  }
  Partio::write(file_name.c_str(), *parts);
  parts->release();
}

auto mpm3d = []() {
  Program prog(Arch::gpu);

  constexpr int n = 128;  // grid_resolution
  const real dt = 1e-4_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto E = 1e3_f;
  // real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 *
  // nu));

  int dim = 3;

  auto f32 = DataType::f32;

  Vector particle_x(f32, dim), particle_v(f32, dim);
  Matrix particle_F(f32, dim, dim), particle_C(f32, dim, dim);
  Global(particle_J, f32);

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

  Global(Jp, f32);

  int n_particles = 8192 * 4;

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  layout([&]() {
    auto place = [&](Expr &expr) { root.fixed(p, n_particles).place(expr); };
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_C(i, j));
      }
      place(particle_x(i));
      place(particle_v(i));
    }
    place(particle_J);

    root.fixed({i, j, k}, {n, n, n})
        .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
  });
  // TODO:... replace this

  TC_ASSERT(bit::is_power_of_two(n));

  auto clear_buffer = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), grid_m, [&] {
      grid_v(0)[i, j, k] = 0.0_f;
      grid_v(1)[i, j, k] = 0.0_f;
      grid_v(2)[i, j, k] = 0.0_f;
      grid_m[i, j, k] = 0.0_f;
    });
  });

  auto p2g = kernel([&]() {
    Declare(p);
    For(p, particle_x(0), [&] {
      auto x = particle_x[p];
      auto v = particle_v[p];
      // auto F = particle_F[p];
      auto C = particle_C[p];
      auto J = particle_J[p];

      auto base_coord = floor(Expr(inv_dx) * x - Expr(0.5_f));
      auto fx = x * Expr(inv_dx) - base_coord;

      Vector w[3];
      w[0] = Expr(0.5_f) * sqr(Expr(1.5_f) - fx);
      w[1] = Expr(0.75_f) - sqr(fx - Expr(1.0_f));
      w[2] = Expr(0.5_f) * sqr(fx - Expr(0.5_f));

      auto cauchy = Expr(E) * (J - Expr(1.0_f));
      auto affine = Expr(particle_mass) * C;
      for (int i = 0; i < dim; i++) {
        affine(i, i) =
            affine(i, i) + Expr(-4 * inv_dx * inv_dx * dt * vol) * cauchy;
      }

      // scatter
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            auto dpos = Vector(dim);
            dpos(0) = dx * ((i * 1.0_f) - fx(0));
            dpos(1) = dx * ((j * 1.0_f) - fx(1));
            dpos(2) = dx * ((k * 1.0_f) - fx(2));
            auto weight = w[i](0) * w[j](1) * w[k](2);
            auto node = (cast<int32>(base_coord(0)) + Expr(i),
                         cast<int32>(base_coord(1)) + Expr(j),
                         cast<int32>(base_coord(2)) + Expr(k));
            grid_v[node] = grid_v[node] +
                           weight * (Expr(particle_mass) * v + affine * dpos);
            grid_m[node] = grid_m[node] + weight * Expr(particle_mass);
          }
        }
      }
    });
  });

  auto cmp_lt = [&](Expr a, Expr b) -> Expr { return a < b; };

  auto grid_op = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), grid_m, [&] {
      auto v0 = load(grid_v[i, j, k](0));
      auto v1 = load(grid_v[i, j, k](1));
      auto v2 = load(grid_v[i, j, k](2));
      auto m = load(grid_m[i, j, k]);

      // auto inv_m = Expr(1.0_f) / max(m, Expr(1e-20_f));
      auto inv_m = Expr(1.0_f) / m;
      auto mask = cmp_lt(Expr(0.0_f), m);
      v0 = select(mask, v0 * inv_m, Expr(0.0_f));
      v1 = select(mask, v1 * inv_m + Expr(dt * -200_f), Expr(0.0_f));
      v2 = select(mask, v2 * inv_m, Expr(0.0_f));

      v0 = select((Expr(n - 5) < i), min(v0, Expr(0.0_f)), v0);
      v1 = select((Expr(n - 5) < j), min(v1, Expr(0.0_f)), v1);
      v2 = select((Expr(n - 5) < k), min(v2, Expr(0.0_f)), v2);

      v0 = select(cmp_lt(i, Expr(5)), max(v0, Expr(0.0_f)), v0);
      v1 = select(cmp_lt(j, Expr(5)), max(v1, Expr(0.0_f)), v1);
      v2 = select(cmp_lt(k, Expr(5)), max(v2, Expr(0.0_f)), v2);

      grid_v[i, j, k](0) = v0;
      grid_v[i, j, k](1) = v1;
      grid_v[i, j, k](2) = v2;
    });
  });

  auto g2p = kernel([&]() {
    Declare(p);
    For(p, particle_x(0), [&] {
      auto x = particle_x[p];
      auto v = Vector(dim);
      // auto F = particle_F[p];
      auto C = Matrix(dim, dim);
      auto J = particle_J[p];

      for (int i = 0; i < dim; i++) {
        v(i) = Expr(0.0_f);
        for (int j = 0; j < dim; j++) {
          C(i, j) = Expr(0.0_f);
        }
      }

      auto base_coord = floor(Expr(inv_dx) * x - Expr(0.5_f));
      auto fx = x * Expr(inv_dx) - base_coord;

      Vector w[3];
      w[0] = Expr(0.5_f) * sqr(Expr(1.5_f) - fx);
      w[1] = Expr(0.75_f) - sqr(fx - Expr(1.0_f));
      w[2] = Expr(0.5_f) * sqr(fx - Expr(0.5_f));

      // scatter
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            auto dpos = Vector(dim);
            dpos(0) = Expr(i * 1.0_f) - fx(0);
            dpos(1) = Expr(j * 1.0_f) - fx(1);
            dpos(2) = Expr(k * 1.0_f) - fx(2);
            auto weight = w[i](0) * w[j](1) * w[k](2);
            auto wv = weight * grid_v[cast<int32>(base_coord(0)) + Expr(i),
                                      cast<int32>(base_coord(1)) + Expr(j),
                                      cast<int32>(base_coord(2)) + Expr(k)];
            v = v + wv;
            C = C + Expr(4 * inv_dx) * outer_product(wv, dpos);
          }
        }
      }

      J = J * (Expr(1.0_f) + Expr(dt) * (C(0, 0) + C(1, 1) + C(2, 2)));
      x = x + Expr(dt) * v;

      particle_C[p] = C;
      particle_v[p] = v;
      particle_J[p] = J;
      particle_x[p] = x;
    });
  });

  int scale = 8;
  GUI gui("MPM", n * scale, n * scale);

  for (int i = 0; i < n_particles; i++) {
    particle_x(0).val<float32>(i) = 0.4_f + rand() * 0.2_f;
    particle_x(1).val<float32>(i) = 0.15_f + rand() * 0.75_f;
    particle_x(2).val<float32>(i) = 0.4_f + rand() * 0.1_f;
    particle_v(1).val<float32>(i) = -0.3_f;
    particle_J.val<float32>(i) = 1_f;
  }

  auto &canvas = gui.get_canvas();

  int frame = 0;
  for (int f = 0; f < 1000; f++) {
    for (int t = 0; t < 50; t++) {
      TC_TIME(clear_buffer());
      TC_TIME(p2g());
      TC_TIME(grid_op());
      TC_TIME(g2p());
    }
    canvas.clear(0x112F41);
    std::vector<Vector3> particles;
    for (int i = 0; i < n_particles; i++) {
      auto x = particle_x(0).val<float32>(i), y = particle_x(1).val<float32>(i);
      auto z = particle_x(2).val<float32>(i);
      particles.push_back(Vector3(x, y, z));
      if (0 < x && x < 1 && 0 < y && y < 1)
        canvas.circle(x, y).radius(2).color(0x068587);
    }

    gui.update();
    write_partio(particles, fmt::format("particles/{:04d}.bgeo", frame));
    frame++;
  }
};
TC_REGISTER_TASK(mpm3d);

TC_NAMESPACE_END
