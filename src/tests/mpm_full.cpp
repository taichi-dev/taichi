#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <Partio.h>
#include <taichi/system/profiler.h>

TLANG_NAMESPACE_BEGIN
std::tuple<Matrix, Matrix, Matrix> sifakis_svd(const Matrix &a);
TLANG_NAMESPACE_END

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
  bool benchmark_dragon = false;
  // Program prog(Arch::x86_64);
  Program prog(Arch::gpu);
  //  prog.config.print_ir = true;
  bool fluid = false;
  bool plastic = true;
  CoreState::set_trigger_gdb_when_crash(true);
  // Program prog(Arch::x86_64);

  constexpr int n = 128;  // grid_resolution
  const real dt = 1e-4_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto E = 2e3_f, nu = 0.3f;
  real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

  constexpr int dim = 3;

  auto f32 = DataType::f32;
  int grid_block_size = 8;
  int particle_block_size = 1;

  Vector particle_x(f32, dim), particle_v(f32, dim);
  Matrix particle_F(f32, dim, dim), particle_C(f32, dim, dim);
  Global(l, i32);
  Global(particle_J, f32);
  Global(gravity_x, f32);

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

  bool sorted = true;
  if (sorted) {
    TC_ASSERT(!fluid);
  }

  int max_n_particles = 1024 * 1024 / 8;

  int n_particles = 0;
  std::vector<float> benchmark_particles(n_particles * 3);
  if (benchmark_dragon) {
    n_particles = 775196;
    auto f = fopen("dragon_particles.bin", "rb");
    std::fread(benchmark_particles.data(), sizeof(float), n_particles * 3, f);
    std::fclose(f);
  } else {
    n_particles = max_n_particles;
  }

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  bool SOA = true;

  layout([&]() {
    SNode *fork;
    if (!SOA)
      fork = &root.dynamic(p, max_n_particles);
    auto place = [&](Expr &expr) {
      if (SOA) {
        if (particle_block_size == 1) {
          root.dynamic(p, max_n_particles).place(expr);
        } else {
          TC_ASSERT(max_n_particles % particle_block_size == 0);
          root.dense(p, max_n_particles / particle_block_size)
              .dense(p, particle_block_size)
              .place(expr);
        }
      } else {
        fork->place(expr);
      }
    };
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_C(i, j));
        place(particle_F(i, j));
      }
      place(particle_x(i));
      place(particle_v(i));
    }
    place(particle_J);

    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, n / grid_block_size);
    block.dense({i, j, k}, grid_block_size)
        .place(grid_v(0), grid_v(1), grid_v(2), grid_m);

    block.dynamic(p, pow<dim>(grid_block_size) * 16).place(l);

    root.place(gravity_x);
  });

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

  auto p2g_naive = kernel([&]() {
    Declare(p);
    if (particle_block_size == 1)
      BlockDim(256);
    For(p, particle_x(0), [&] {
      auto x = particle_x[p];
      auto v = particle_v[p];
      auto C = particle_C[p];

      Expr J;
      Matrix F;
      if (fluid) {
        J = particle_J[p] * (1.0_f + dt * (C(0, 0) + C(1, 1) + C(2, 2)));
        particle_J[p] = J;
      } else {
        F = Eval((Matrix::identity(3) + dt * C) * particle_F[p]);
      }
      Mutable(F, DataType::f32);

      auto base_coord = floor(Expr(inv_dx) * x - Expr(0.5_f));
      auto fx = x * Expr(inv_dx) - base_coord;

      Vector w[] = {Eval(0.5_f * sqr(1.5_f - fx)),
                    Eval(0.75_f - sqr(fx - 1.0_f)),
                    Eval(0.5_f * sqr(fx - 0.5_f))};

      Matrix cauchy(3, 3);
      Local(mu) = mu_0;
      Local(lambda) = lambda_0;
      if (fluid) {
        cauchy = (J - 1.0_f) * Matrix::identity(3) * E;
      } else {
        auto svd = sifakis_svd(F);
        auto R = std::get<0>(svd) * transposed(std::get<2>(svd));
        auto sig = std::get<1>(svd);
        Mutable(sig, DataType::f32);
        auto oldJ = sig(0) * sig(1) * sig(2);
        if (plastic) {
          for (int i = 0; i < dim; i++) {
            sig(i) = clamp(sig(i), 1 - 2.5e-2f, 1 + 7.5e-3f);
          }
          auto newJ = sig(0) * sig(1) * sig(2);
          // plastic J
          auto Jp = particle_J[p] * oldJ / newJ;
          particle_J[p] = Jp;
          J = newJ;
          F = std::get<0>(svd) * diag_matrix(sig) *
              transposed(std::get<2>(svd));
          particle_F[p] = F;
          /*
          auto harderning = exp((1.0f - Jp) * 10.0f);
          mu *= harderning;
          lambda *= harderning;
          */
        } else {
          J = oldJ;
          particle_F[p] = F;
        }
        cauchy = Eval(2.0_f * mu * (F - R) * transposed(F) +
                      (Matrix::identity(3) * lambda) * (J - 1.0f) * J);
      }

      auto affine = Expr(particle_mass) * C +
                    Expr(-4 * inv_dx * inv_dx * dt * vol) * cauchy;

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
            Atomic(grid_v[node]) +=
                weight * (Expr(particle_mass) * v + affine * dpos);
            Atomic(grid_m[node]) += weight * Expr(particle_mass);
          }
        }
      }
    });
  });

  auto clear_lists = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    BlockDim(256);
    For((i, j, k), l.parent(), [&] { Clear(l.parent(), (i, j, k)); });
  });

  auto sort = kernel([&] {
    Declare(p);
    BlockDim(256);
    For(p, particle_x(0), [&] {
      auto node_coord = floor(particle_x[p] * inv_dx - 0.5_f);
      Activate(l.parent(), (node_coord(0), node_coord(1), node_coord(2)));
      Append(l.parent(),
             (cast<int32>(node_coord(0)), cast<int32>(node_coord(1)),
              cast<int32>(node_coord(2))),
             p);
    });
  });

  auto p2g_sorted = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    Declare(p_ptr);
    BlockDim(256);
    For((i, j, k, p_ptr), l, [&] {
      auto p = Eval(l[i, j, k, p_ptr]);
      /*
      auto x = particle_x[p];
      Print(Probe(l.parent(), (i, j, k)));
      auto dx = x(0) * inv_dx - 0.5f - cast<float32>(i);
      auto dy = x(1) * inv_dx - 0.5f - cast<float32>(j);
      auto dz = x(2) * inv_dx - 0.5f - cast<float32>(k);
      auto max_d = max(dx, max(dy, dz));
      auto min_d = min(dx, min(dy, dz));
      If(min_d < 0.0f, [&] { Print(min_d); });
      If(max_d > 1.0f * grid_block_size, [&] { Print(max_d); });
      */

      auto x = particle_x[p];
      auto v = particle_v[p];
      auto C = particle_C[p];

      Expr J;
      Matrix F;
      if (fluid) {
        J = particle_J[p] * (1.0_f + dt * (C(0, 0) + C(1, 1) + C(2, 2)));
        particle_J[p] = J;
      } else {
        F = Eval((Matrix::identity(3) + dt * C) * particle_F[p]);
      }
      Mutable(F, DataType::f32);

      auto base_coord = floor(Expr(inv_dx) * x - Expr(0.5_f));
      auto fx = x * Expr(inv_dx) - base_coord;

      Vector w[] = {Eval(0.5_f * sqr(1.5_f - fx)),
                    Eval(0.75_f - sqr(fx - 1.0_f)),
                    Eval(0.5_f * sqr(fx - 0.5_f))};

      Matrix cauchy(3, 3);
      Local(mu) = mu_0;
      Local(lambda) = lambda_0;
      if (fluid) {
        cauchy = (J - 1.0_f) * Matrix::identity(3) * E;
      } else {
        auto svd = sifakis_svd(F);
        auto R = std::get<0>(svd) * transposed(std::get<2>(svd));
        auto sig = std::get<1>(svd);
        Mutable(sig, DataType::f32);
        auto oldJ = sig(0) * sig(1) * sig(2);
        if (plastic) {
          for (int i = 0; i < dim; i++) {
            sig(i) = clamp(sig(i), 1 - 2.5e-2f, 1 + 7.5e-3f);
          }
          auto newJ = sig(0) * sig(1) * sig(2);
          // plastic J
          auto Jp = particle_J[p] * oldJ / newJ;
          particle_J[p] = Jp;
          J = newJ;
          F = std::get<0>(svd) * diag_matrix(sig) *
              transposed(std::get<2>(svd));
          particle_F[p] = F;
          /*
          auto harderning = exp((1.0f - Jp) * 10.0f);
          mu *= harderning;
          lambda *= harderning;
          */
        } else {
          J = oldJ;
          particle_F[p] = F;
        }
        cauchy = Eval(2.0_f * mu * (F - R) * transposed(F) +
                      (Matrix::identity(3) * lambda) * (J - 1.0f) * J);
      }

      auto affine = Expr(particle_mass) * C +
                    Expr(-4 * inv_dx * inv_dx * dt * vol) * cauchy;

      auto base_coord_i = cast<int32>(base_coord(0));
      auto base_coord_j = cast<int32>(base_coord(1));
      auto base_coord_k = cast<int32>(base_coord(2));

      // scatter
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          for (int c = 0; c < 3; c++) {
            auto dpos = Vector(dim);
            dpos(0) = dx * ((a * 1.0_f) - fx(0));
            dpos(1) = dx * ((b * 1.0_f) - fx(1));
            dpos(2) = dx * ((c * 1.0_f) - fx(2));
            auto weight = w[a](0) * w[b](1) * w[c](2);
            auto node = (base_coord_i + Expr(a), base_coord_j + Expr(b),
                         base_coord_k + Expr(c));
            Atomic(grid_v[node]) +=
                weight * (Expr(particle_mass) * v + affine * dpos);
            Atomic(grid_m[node]) += weight * Expr(particle_mass);
          }
        }
      }
    });
  });

  auto p2g = [&] {
    if (sorted) {
      clear_lists();
      sort();
      p2g_sorted();
    } else {
      p2g_naive();
    }
  };

  auto grid_op = kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), grid_m, [&] {
      Local(v0) = grid_v[i, j, k](0);
      Local(v1) = grid_v[i, j, k](1);
      Local(v2) = grid_v[i, j, k](2);
      auto m = load(grid_m[i, j, k]);

      If(m > 0.0f, [&]() {
        auto inv_m = Eval(1.0f / m);
        v0 *= inv_m;
        v1 *= inv_m;
        v2 *= inv_m;

        auto f = gravity_x[Expr(0)];
        v1 += dt * (-100_f + abs(f));
        v0 += dt * f;
      });

      v0 = select(n - 5 < i, min(v0, Expr(0.0_f)), v0);
      v1 = select(n - 5 < j, min(v1, Expr(0.0_f)), v1);
      v2 = select(n - 5 < k, min(v2, Expr(0.0_f)), v2);

      v0 = select(i < 5, max(v0, Expr(0.0_f)), v0);
      v1 = select(j < 5, max(v1, Expr(0.0_f)), v1);
      v2 = select(k < 5, max(v2, Expr(0.0_f)), v2);

      grid_v[i, j, k](0) = v0;
      grid_v[i, j, k](1) = v1;
      grid_v[i, j, k](2) = v2;
    });
  });

  auto g2p = kernel([&]() {
    Declare(p);
    if (particle_block_size == 1)
      BlockDim(256);
    For(p, particle_x(0), [&] {
      auto x = particle_x[p];
      auto v = Vector(dim);
      Mutable(v, DataType::f32);
      // auto F = particle_F[p];
      auto C = Matrix(dim, dim);
      Mutable(C, DataType::f32);

      for (int i = 0; i < dim; i++) {
        v(i) = Expr(0.0_f);
        for (int j = 0; j < dim; j++) {
          C(i, j) = Expr(0.0_f);
        }
      }

      auto base_coord = floor(inv_dx * x - 0.5_f);
      auto fx = x * Expr(inv_dx) - base_coord;

      Vector w[] = {Eval(0.5_f * sqr(1.5_f - fx)),
                    Eval(0.75_f - sqr(fx - 1.0_f)),
                    Eval(0.5_f * sqr(fx - 0.5_f))};

      auto base_i = cast<int32>(base_coord(0));
      auto base_j = cast<int32>(base_coord(1));
      auto base_k = cast<int32>(base_coord(2));

      // scatter
      for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
          for (int r = 0; r < 3; r++) {
            auto dpos = Vector(dim);
            dpos(0) = Expr(p * 1.0_f) - fx(0);
            dpos(1) = Expr(q * 1.0_f) - fx(1);
            dpos(2) = Expr(r * 1.0_f) - fx(2);
            auto weight = w[p](0) * w[q](1) * w[r](2);
            auto wv =
                weight *
                grid_v[base_i + Expr(p), base_j + Expr(q), base_k + Expr(r)];
            v = v + wv;
            C = C + Expr(4 * inv_dx) * outer_product(wv, dpos);
          }
        }
      }

      x = x + dt * v;

      particle_C[p] = C;
      particle_v[p] = v;
      particle_x[p] = x;
    });
  });

  std::vector<int> index(n_particles);
  for (int i = 0; i < n_particles; i++) {
    index[i] = i;
  }
  // std::random_shuffle(index.begin(), index.end());

  auto reset = [&] {
    for (int i = 0; i < n_particles; i++) {
      if (benchmark_dragon) {
        for (int d = 0; d < dim; d++) {
          particle_x(d).val<float32>(i) =
              benchmark_particles[dim * index[i] + d];
        }
      } else {
        particle_x(0).val<float32>(i) = 0.4_f + rand() * 0.2_f;
        particle_x(1).val<float32>(i) = 0.15_f + rand() * 0.75_f;
        particle_x(2).val<float32>(i) = 0.4_f + rand() * 0.2_f;
      }
      particle_v(0).val<float32>(i) = 0._f;
      particle_v(1).val<float32>(i) = -0.3_f;
      particle_v(2).val<float32>(i) = 0._f;
      particle_J.val<float32>(i) = 1_f;
      if (!fluid) {
        for (int p = 0; p < dim; p++) {
          for (int q = 0; q < dim; q++) {
            particle_F(p, q).val<float32>(i) = (p == q);
          }
        }
      }
    }
  };

  reset();

  int scale = 128 * 6 / n;
  GUI gui("MPM", n * scale + 200, n * scale);
  int angle = 0;
  int gravity_x_slider = 0;
  gui.button("Restart", reset)
      .slider("Camera", angle, 0, 360, 1)
      .slider("Gravity Dir", gravity_x_slider, -100, 100);

  auto &canvas = gui.get_canvas();

  int frame = 0;
  for (int f = 0;; f++) {
    TC_PROFILER("mpm 3d");
    for (int t = 0; t < 20; t++) {
      TC_PROFILE("reset grid", clear_buffer());
      TC_PROFILE("p2g", p2g());
      TC_PROFILE("grid_op", grid_op());
      TC_PROFILE("g2p", g2p());
    }
    canvas.clear(0x112F41);

    Matrix4 trans(1);
    trans = matrix4_translate(&trans, Vector3(-0.5f));
    trans = matrix4_scale_s(&trans, 0.7f);
    trans = matrix4_rotate_angle_axis(&trans, angle * 1.0f, Vector3::axis(1));
    trans = matrix4_rotate_angle_axis(&trans, 15.0f, Vector3::axis(0));
    trans = matrix4_translate(&trans, Vector3(0.5f));

    {
      TC_PROFILER("cpu render");

      std::vector<Vector3> particles;
      for (int i = 0; i < n_particles; i++) {
        auto x = particle_x(0).val<float32>(i),
             y = particle_x(1).val<float32>(i);
        auto z = particle_x(2).val<float32>(i);

        particles.push_back(Vector3(x, y, z));

        Vector3 pos(x, y, z);
        pos = transform(trans, pos);

        if (0.01f < pos.x && pos.x < 0.99f && 0.01f < pos.y && pos.y < 0.99f)
          canvas.circle(pos.x, pos.y)
              .radius(1.6)
              .color(0.4f + 0.6f * x, 0.4f + 0.6f * y, 0.4f + 0.6f * z, 1.0f);
      }

      gravity_x.val<float32>() = gravity_x_slider;
      gui.update();
      // write_partio(particles, fmt::format("particles/{:04d}.bgeo", frame));
      frame++;
    }

    print_profile_info();
  }
};
TC_REGISTER_TASK(mpm3d);

TC_NAMESPACE_END
