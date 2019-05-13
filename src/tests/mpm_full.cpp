#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <Partio.h>
#include <taichi/system/profiler.h>
#include "svd.h"
#include "volume_renderer.h"

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

enum class MPMMaterial : int { fluid, jelly, snow, sand };

auto mpm3d = []() {
  Program prog(Arch::gpu);
  prog.config.lower_access = false;

  TRenderer renderer;

  CoreState::set_trigger_gdb_when_crash(true);

  bool benchmark_dragon = false;
  // Program prog(Arch::x86_64);
  // prog.config.print_ir = true;
  auto material = MPMMaterial::snow;
  constexpr int dim = 3;
  constexpr bool highres = true;

  constexpr int n = highres ? 256 : 128;  // grid_resolution
  constexpr int grid_n = n * 4;
  const real dt = 1e-5_f * 256 / n, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto E = 1e4_f, nu = 0.3f;
  real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

  auto friction_angle = 45._f;
  real sin_phi = std::sin(friction_angle / 180._f * real(3.141592653));
  auto alpha = std::sqrt(2._f / 3._f) * 2._f * sin_phi / (3._f - sin_phi);

  auto f32 = DataType::f32;
  int grid_block_size = 4;

  Vector particle_x("x", f32, dim), particle_v("v", f32, dim);
  Matrix particle_F("F", f32, dim, dim), particle_C("C", f32, dim, dim);

  NamedScalar(l, l, i32);
  NamedScalar(particle_J, J, f32);
  NamedScalar(gravity_x, g, f32);

  Vector grid_v("v^{g}", f32, dim);
  NamedScalar(grid_m, m ^ {p}, f32);

  int max_n_particles = 1024 * 1024;

  int n_particles = 0;
  std::vector<float> benchmark_particles;
  std::vector<Vector3> p_x;
  if (benchmark_dragon) {
    n_particles = 775196;
    p_x.resize(n_particles);
    TC_ASSERT(n_particles <= max_n_particles);
    auto f = fopen("dragon_particles.bin", "rb");
    TC_ASSERT_INFO(f, "./dragon_particles.bin not found");
    benchmark_particles.resize(n_particles * 3);
    std::fread(benchmark_particles.data(), sizeof(float), n_particles * 3, f);
    std::fclose(f);
    for (int i = 0; i < n_particles; i++) {
      for (int j = 0; j < dim; j++)
        p_x[i][j] = benchmark_particles[i * 3 + j];
    }
  } else {
    n_particles = max_n_particles / (highres ? 1 : 8);
    p_x.resize(n_particles);
    if (false) {
      for (int i = 0; i < n_particles; i++) {
        p_x[i].x = 0.4_f + rand() * 0.2_f;
        p_x[i].y = 0.15_f + rand() * 0.55_f;
        p_x[i].z = 0.4_f + rand() * 0.2_f;
      }
    } else {
      for (int i = 0; i < n_particles; i++) {
        Vector3 offset = Vector3::rand() - Vector3(0.5_f);
        while (offset.length() > 0.5f) {
          offset = Vector3::rand() - Vector3(0.5_f);
        }
        p_x[i] = Vector3(0.5_f) + offset * 0.25f;
      }
    }
  }

  TC_ASSERT(n_particles <= max_n_particles);

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  bool particle_soa = false;

  layout([&]() {
    SNode *fork;
    if (!particle_soa)
      fork = &root.dynamic(p, max_n_particles);
    auto place = [&](Expr &expr) {
      if (particle_soa) {
        root.dynamic(p, max_n_particles).place(expr);
      } else {
        fork->place(expr);
      }
    };
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_F(i, j));
      }
    }
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        place(particle_C(i, j));
      }
    }
    for (int i = 0; i < dim; i++) {
      place(particle_x(i));
    }
    for (int i = 0; i < dim; i++)
      place(particle_v(i));
    place(particle_J);

    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, grid_n / 4 / grid_block_size)
                      .pointer()
                      .dense({i, j, k}, 4)
                      .pointer();
    constexpr bool block_soa = true;

    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size)
          .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
    }

    block.dynamic(p, pow<dim>(grid_block_size) * 64).place(l);

    root.place(gravity_x);

    renderer.place_data();
  });

  renderer.declare_kernels();

  // prog.visualize_layout("layout.tex");

  TC_ASSERT(bit::is_power_of_two(n));

  Kernel(sort).def([&] {
    BlockDim(512);
    For(particle_x(0), [&](Expr p) {
      auto node_coord = floor(particle_x[p] * inv_dx - 0.5_f);
      Append(l.parent(),
             (cast<int32>(node_coord(0)), cast<int32>(node_coord(1)),
              cast<int32>(node_coord(2))),
             p);
    });
  });

  auto project = [&](Vector sigma, const Expr &p) {
    real fdim = dim;
    auto sigma_out = Var(Vector(dim));
    auto epsilon = Var(Vector(dim));
    for (int i = 0; i < dim; i++) {
      epsilon(i) = log(max(abs(sigma(i)), 1e-4_f));
      sigma_out(i) = 1.0_f;
    }
    auto tr = Var(epsilon.sum() + particle_J[p]);
    auto epsilon_hat = Var(epsilon - tr / fdim);
    auto epsilon_hat_norm = Var(epsilon_hat.norm() + 1e-20_f);
    If(tr >= 0.0_f).Then([&] { particle_J[p] += epsilon.sum(); }).Else([&] {
      particle_J[p] = 0.0f;
      auto delta_gamma =
          Var(epsilon_hat_norm +
              (fdim * lambda_0 + 2.0_f * mu_0) / (2.0_f * mu_0) * tr * alpha);
      sigma_out = exp(epsilon -
                      max(0.0_f, delta_gamma) / epsilon_hat_norm * epsilon_hat);
    });

    return sigma_out;
  };

  Kernel(p2g_sorted).def([&] {
    BlockDim(128);

    Cache(0, grid_v(0));
    Cache(0, grid_v(1));
    Cache(0, grid_v(2));
    Cache(0, grid_m);
    For(l, [&](Expr i, Expr j, Expr k, Expr p_ptr) {
      auto p = Var(l[i, j, k, p_ptr]);

      auto x = particle_x[p];
      auto v = particle_v[p];
      auto C = particle_C[p];

      Expr J;
      Matrix F;
      if (material == MPMMaterial::fluid) {
        particle_J[p] *= 1.0_f + dt * (C(0, 0) + C(1, 1) + C(2, 2));
      } else {
        F = Var(Matrix::identity(dim) + dt * C) * particle_F[p];
      }

      auto base_coord = floor(inv_dx * x - 0.5_f);
      auto fx = x * inv_dx - base_coord;

      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};

      Matrix cauchy(3, 3);
      auto mu = Var(mu_0);
      auto lambda = Var(lambda_0);
      if (material == MPMMaterial::fluid) {
        cauchy = (J - 1.0_f) * Matrix::identity(3) * E;
      } else if (material == MPMMaterial::jelly ||
                 material == MPMMaterial::snow) {
        auto svd = sifakis_svd(F);
        auto R = std::get<0>(svd) * transposed(std::get<2>(svd));
        auto sig = Var(std::get<1>(svd));
        auto oldJ = Var(sig(0) * sig(1) * sig(2));
        if (material == MPMMaterial::snow) {
          for (int d = 0; d < dim; d++)
            sig(d) = clamp(sig(d), 1 - 2.5e-2f, 1 + 7.5e-3f);
          auto newJ = sig(0) * sig(1) * sig(2);
          auto Jp = Var(clamp(particle_J[p] * oldJ / newJ, 0.6_f, 20.0_f));
          J = newJ;
          F = std::get<0>(svd) * diag_matrix(sig) *
              transposed(std::get<2>(svd));
          auto harderning = exp((1.0f - Jp) * 10.0f);
          mu *= harderning;
          lambda *= harderning;
          particle_J[p] = Jp;
        } else {
          J = oldJ;
        }
        cauchy = 2.0_f * mu * (F - R) * transposed(F) +
                 (Matrix::identity(3) * lambda) * (J - 1.0f) * J;
      } else if (material == MPMMaterial::sand) {
        auto svd = sifakis_svd(F);
        auto u = std::get<0>(svd), sig = Var(std::get<1>(svd)),
             v = std::get<2>(svd);
        sig = project(std::get<1>(svd), p);
        F = u * diag_matrix(sig) * transposed(v);
        auto log_sig = log(sig);
        auto inv_sig = 1.0_f / sig;
        auto center = Var(2.0_f * mu_0 * inv_sig.element_wise_prod(log_sig) +
                          lambda_0 * log_sig.sum() * inv_sig);
        cauchy = u * diag_matrix(center) * transposed(v) * transposed(F);
      }

      if (material != MPMMaterial::fluid)
        particle_F[p] = F;

      auto affine = Expr(particle_mass) * C +
                    Expr(-4 * inv_dx * inv_dx * dt * vol) * cauchy;

      int low = 0, high = 1;
      auto base_coord_i =
          AssumeInRange(cast<int32>(base_coord(0)), i, low, high);
      auto base_coord_j =
          AssumeInRange(cast<int32>(base_coord(1)), j, low, high);
      auto base_coord_k =
          AssumeInRange(cast<int32>(base_coord(2)), k, low, high);

      Assert(base_coord_i < i + 4);
      Assert(base_coord_i - i >= 0);
      Assert(base_coord_j < j + 4);
      Assert(base_coord_j - j >= 0);
      Assert(base_coord_k < k + 4);
      Assert(base_coord_k - k >= 0);
      Assert(i % 4 == 0);
      Assert(j % 4 == 0);
      Assert(k % 4 == 0);

      // scatter
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          for (int c = 0; c < 3; c++) {
            auto offset = Vector({a, b, c});
            auto dpos = dx * (Vector({a, b, c}).cast_elements<float32>() - fx);
            auto weight = w[a](0) * w[b](1) * w[c](2);
            auto node =
                Vector({base_coord_i, base_coord_j, base_coord_k}) + offset;
            Atomic(grid_v[node]) +=
                weight * (particle_mass * v + affine * dpos);
            Atomic(grid_m[node]) += weight * particle_mass;
          }
        }
      }
    });
  });

  auto check_fluctuation = [&] {
    int last_nb = -1;
    while (1) {
      grid_m.parent().parent().snode()->clear_data_and_deactivate();
      sort();
      p2g_sorted();
      auto stat = grid_m.parent().parent().snode()->stat();
      int nb = stat.num_resident_blocks;
      TC_P(nb);
      if (last_nb == -1) {
        last_nb = nb;
      } else {
        TC_ASSERT(last_nb == nb);
      }
    }
  };

  auto p2g = [&] {
    // check_fluctuation();
    grid_m.parent().parent().snode()->clear_data();
    sort();
    p2g_sorted();
  };

  Kernel(grid_op).def([&]() {
    For(grid_m, [&](Expr i, Expr j, Expr k) {
      auto v = Var(grid_v[i, j, k]);
      auto m = load(grid_m[i, j, k]);

      int bound = 8;

      If(m > 0.0f, [&]() {
        auto inv_m = Var(1.0f / m);
        v *= inv_m;

        auto f = gravity_x[Expr(0)];
        v(1) += dt * (-1000_f + abs(f));
        v(0) += dt * f;
      });

      /*
      v(0) = select(n - bound < i, min(v(0), Expr(0.0_f)), v(0));
      v(1) = select(n - bound < j, min(v(1), Expr(0.0_f)), v(1));
      v(2) = select(n - bound < k, min(v(2), Expr(0.0_f)), v(2));

      v(0) = select(i < bound, max(v(0), Expr(0.0_f)), v(0));
      v(2) = select(k < bound, max(v(2), Expr(0.0_f)), v(2));
      */

      If(j < bound, [&] {
        auto norm = Var(sqrt(v(0) * v(0) + v(2) * v(2)));
        auto s = Var(clamp(
            (norm + v(1) * (material == MPMMaterial::sand ? 1.0f : 0.10f)) /
                (norm + 1e-30f),
            Expr(0.0_f), Expr(1.0_f)));

        v *= s;
        v(1) = max(v(1), Expr(0.0_f));
      });

      grid_v[i, j, k] = v;
    });
  });

  Kernel(g2p).def([&]() {
    BlockDim(128);
    Cache(0, grid_v(0));
    Cache(0, grid_v(1));
    Cache(0, grid_v(2));
    For(l, [&](Expr i, Expr j, Expr k, Expr p_ptr) {
      auto p = Var(l[i, j, k, p_ptr]);
      Assert(p >= 0);
      Assert(p < n_particles);
      auto x = Var(particle_x[p]);
      auto v = Var(Vector(dim));
      auto C = Var(Matrix(dim, dim));

      for (int i = 0; i < dim; i++) {
        v(i) = Expr(0.0_f);
        for (int j = 0; j < dim; j++) {
          C(i, j) = Expr(0.0_f);
        }
      }

      auto base_coord = floor(inv_dx * x - 0.5_f);
      auto fx = x * inv_dx - base_coord;

      Vector w[] = {Var(0.5_f * sqr(1.5_f - fx)), Var(0.75_f - sqr(fx - 1.0_f)),
                    Var(0.5_f * sqr(fx - 0.5_f))};

      int low = 0, high = 1;
      auto base_coord_i =
          AssumeInRange(cast<int32>(base_coord(0)), i, low, high);
      auto base_coord_j =
          AssumeInRange(cast<int32>(base_coord(1)), j, low, high);
      auto base_coord_k =
          AssumeInRange(cast<int32>(base_coord(2)), k, low, high);

      Assert(base_coord_i < i + 4);
      Assert(base_coord_i - i >= 0);
      Assert(base_coord_j < j + 4);
      Assert(base_coord_j - j >= 0);
      Assert(base_coord_k < k + 4);
      Assert(base_coord_k - k >= 0);
      Assert(i % 4 == 0);
      Assert(j % 4 == 0);
      Assert(k % 4 == 0);

      for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
          for (int r = 0; r < 3; r++) {
            auto dpos = Vector({p, q, r}).cast_elements<float32>() - fx;
            auto weight = w[p](0) * w[q](1) * w[r](2);
            auto wv =
                weight *
                grid_v[base_coord_i + p, base_coord_j + q, base_coord_k + r];
            v += wv;
            C += outer_product(wv, dpos);
          }
        }
      }

      particle_C[p] = (4 * inv_dx) * C;
      particle_v[p] = v;
      particle_x[p] = x + dt * v;
    });
  });

  auto block_id = [&](Vector3 x) {
    auto xi = (x * inv_dx - Vector3(0.5f)).floor().template cast<int>() /
              Vector3i(grid_block_size);
    return xi.x * pow<2>(n / grid_block_size) + xi.y * n / grid_block_size +
           xi.z;
  };

  std::sort(p_x.begin(), p_x.end(),
            [&](Vector3 a, Vector3 b) { return block_id(a) < block_id(b); });

  auto reset = [&] {
    for (int i = 0; i < n_particles; i++) {
      for (int d = 0; d < dim; d++) {
        particle_x(d).val<float32>(i) = p_x[i][d];
      }
      particle_v(0).val<float32>(i) = 0._f;
      particle_v(1).val<float32>(i) = -3.0_f;
      particle_v(2).val<float32>(i) = 0._f;
      if (material == MPMMaterial::sand) {
        particle_J.val<float32>(i) = 0_f;
      } else {
        particle_J.val<float32>(i) = 1_f;
      }
      if (material != MPMMaterial::fluid) {
        for (int p = 0; p < dim; p++) {
          for (int q = 0; q < dim; q++) {
            particle_F(p, q).val<float32>(i) = (p == q);
          }
        }
      }
    }
  };

  reset();

  Vector2i cam_res(1024, 512);

  int np = 512;
  GUI gui("MPM", cam_res);

  auto simulate_frame = [&]() {
    grid_m.parent().parent().snode()->clear_data_and_deactivate();
    grid_m.parent()
        .parent()
        .parent()
        .parent()
        .snode()
        ->clear_data_and_deactivate();
    auto t = Time::get_time();
    for (int f = 0; f < 200; f++) {
      TC_PROFILE("p2g", p2g());
      TC_PROFILE("grid_op", grid_op());
      TC_PROFILE("g2p", g2p());
    }
    prog.profiler_print();
    auto ms_per_substep = (Time::get_time() - t) / 200 * 1000;
    TC_P(ms_per_substep);
  };

  Kernel(set_renderer_volume).def([&] {
    For(grid_m, [&](Expr i, Expr j, Expr k) {
      If (grid_m[i, j, k] > 0.0f).Then([&]{
        renderer.density[i, j, k] = grid_m[i, j, k] * 50.0f;
      });
    });
  });

  auto tone_map = [](real x) { return std::sqrt(x); };
  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    simulate_frame();
    auto res = canvas.img.get_res();

    renderer.density.parent().parent().snode()->clear_data_and_deactivate();
    renderer.density.parent().parent().parent().snode()->clear_data_and_deactivate();
    renderer.buffer(0).parent().snode()->clear_data();
    set_renderer_volume();
    renderer.preprocess_volume();
    int nsamples = 10;
    for (int s = 0; s < nsamples; s++) {
      renderer.sample();
    }

    real scale = 1.0f / nsamples;
    for (int i = 0; i < np * np * 2; i++) {
      gui.canvas->img[i / np][i % np] =
          Vector4(tone_map(scale * renderer.buffer(0).val<float32>(i)),
                  tone_map(scale * renderer.buffer(1).val<float32>(i)),
                  tone_map(scale * renderer.buffer(2).val<float32>(i)), 1);
    }

    prog.profiler_print();

    gui.update();
    auto render_dir = fmt::format("{}_rendered", "mpm");
    create_directories(render_dir);
    gui.get_canvas().img.write_as_image(
        fmt::format("{}/{:05d}.png", render_dir, frame));
    print_profile_info();
  }
};
TC_REGISTER_TASK(mpm3d);

TC_NAMESPACE_END
