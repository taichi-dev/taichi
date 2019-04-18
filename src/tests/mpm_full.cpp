#include "../tlang.h"
#include <taichi/util.h>
#include <taichi/visual/gui.h>
#include <taichi/common/bit.h>
#include <Partio.h>
#include <taichi/system/profiler.h>
#include <taichi/visualization/particle_visualization.h>

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

auto reset_grid_benchmark = []() {
  Program prog(Arch::gpu);
  prog.config.print_ir = true;

  constexpr int n = 256;  // grid_resolution
  constexpr int dim = 3;

  auto f32 = DataType::f32;
  int grid_block_size = 4;

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

  auto i = Index(0), j = Index(1), k = Index(2);

  layout([&]() {
    TC_ASSERT(n % grid_block_size == 0);
    auto &block = root.dense({i, j, k}, n / grid_block_size);
    constexpr bool block_soa = false;
    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size)
          .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
      //.place(grid_m);
    }
  });

  TC_ASSERT(bit::is_power_of_two(n));

  auto &reset_grid = kernel([&]() {
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

  while (1)
    TC_TIME(reset_grid());
};
TC_REGISTER_TASK(reset_grid_benchmark);

auto mpm3d = []() {
  bool benchmark_dragon = false;
  Program prog(Arch::gpu);
  // Program prog(Arch::x86_64);
  // prog.config.print_ir = true;
  bool fluid = false;
  bool plastic = true;
  constexpr bool highres = true;
  CoreState::set_trigger_gdb_when_crash(true);

  constexpr int n = highres ? 256 : 128;  // grid_resolution
  const real dt = 2e-5_f * 256 / n, dx = 1.0_f / n, inv_dx = 1.0_f / dx;
  auto particle_mass = 1.0_f, vol = 1.0_f;
  auto E = 3e4_f, nu = 0.3f;
  real mu_0 = E / (2 * (1 + nu)), lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

  constexpr int dim = 3;

  auto f32 = DataType::f32;
  int grid_block_size = 4;
  int particle_block_size = 1;

  Vector particle_x(f32, dim), particle_v(f32, dim);
  Matrix particle_F(f32, dim, dim), particle_C(f32, dim, dim);
  Global(l, i32);
  Global(particle_J, f32);
  Global(gravity_x, f32);

  Vector grid_v(f32, dim);
  Global(grid_m, f32);

  bool sorted = true;

  int max_n_particles = 1024 * 1024;

  int n_particles = 0;
  std::vector<float> benchmark_particles;
  if (benchmark_dragon) {
    n_particles = 775196;
    auto f = fopen("dragon_particles.bin", "rb");
    TC_ASSERT(f);
    benchmark_particles.resize(n_particles * 3);
    std::fread(benchmark_particles.data(), sizeof(float), n_particles * 3, f);
    std::fclose(f);
  } else {
    n_particles = max_n_particles / (highres ? 1 : 8);
  }

  TC_ASSERT(n_particles <= max_n_particles);

  auto i = Index(0), j = Index(1), k = Index(2);
  auto p = Index(3);

  bool SOA = false;

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
    auto &block = root.dense({i, j, k}, n / grid_block_size).pointer();
    constexpr bool block_soa = false;

    if (block_soa) {
      block.dense({i, j, k}, grid_block_size).place(grid_v(0));
      block.dense({i, j, k}, grid_block_size).place(grid_v(1));
      block.dense({i, j, k}, grid_block_size).place(grid_v(2));
      block.dense({i, j, k}, grid_block_size).place(grid_m);
    } else {
      block.dense({i, j, k}, grid_block_size)
          .place(grid_v(0), grid_v(1), grid_v(2), grid_m);
    }

    block.dynamic(p, pow<dim>(grid_block_size) * 32).place(l);

    root.place(gravity_x);
  });

  TC_ASSERT(bit::is_power_of_two(n));

  kernel([&]() {
    Declare(i);
    Declare(j);
    Declare(k);
    int t = n / grid_block_size;
    For(i, 0, t, [&] {
      For(j, 0, t, [&] {
        For(k, 0, t, [&] {
          Activate(grid_m, (i * grid_block_size, j * grid_block_size,
                            k * grid_block_size));
        });
      });
    });
  })();

  Kernel(reset_grid).def([&]() {
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

  Kernel(p2g_naive).def([&]() {
    Declare(p);
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
        F = Eval((Matrix::identity(dim) + dt * C) * particle_F[p]);
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
        auto oldJ = Eval(sig(0) * sig(1) * sig(2));
        if (plastic) {
          for (int i = 0; i < dim; i++) {
            sig(i) = clamp(sig(i), 1 - 2.5e-2f, 1 + 7.5e-3f);
          }
          auto newJ = Eval(sig(0) * sig(1) * sig(2));
          // plastic J
          auto Jp = particle_J[p] * oldJ / newJ;
          J = newJ;
          F = std::get<0>(svd) * diag_matrix(sig) *
              transposed(std::get<2>(svd));
          auto hardening = exp((1.0f - Jp) * 10.0f);

          mu *= hardening;
          lambda *= hardening;
          particle_J[p] = Jp;
        } else {
          J = oldJ;
        }
        particle_F[p] = F;
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

  Kernel(clear_lists).def([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    // BlockDim(64);
    For((i, j, k), l.parent(), [&] { Clear(l.parent(), (i, j, k)); });
  });

  Kernel(sort).def([&] {
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

  Kernel(p2g_sorted).def([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    Declare(p_ptr);
    BlockDim(128);

    Cache(0, grid_v(0));
    Cache(0, grid_v(1));
    Cache(0, grid_v(2));
    Cache(0, grid_m);
    For((i, j, k, p_ptr), l, [&] {
      auto p = Eval(l[i, j, k, p_ptr]);

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
          auto Jp = Eval(particle_J[p] * oldJ / newJ);
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
        particle_F[p] = F;
        cauchy = Eval(2.0_f * mu * (F - R) * transposed(F) +
                      (Matrix::identity(3) * lambda) * (J - 1.0f) * J);
      }

      auto affine = Expr(particle_mass) * C +
                    Expr(-4 * inv_dx * inv_dx * dt * vol) * cauchy;

      int low = 0, high = 1;
      auto base_coord_i =
          Eval(AssumeInRange(cast<int32>(base_coord(0)), i, low, high));
      auto base_coord_j =
          Eval(AssumeInRange(cast<int32>(base_coord(1)), j, low, high));
      auto base_coord_k =
          Eval(AssumeInRange(cast<int32>(base_coord(2)), k, low, high));

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

  Kernel(grid_op).def([&]() {
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

  Kernel(g2p).def([&]() {
    // benchmark_kernel();
    Declare(i);
    Declare(j);
    Declare(k);
    Declare(p_ptr);
    BlockDim(64);

    if (sorted) {
      Cache(0, grid_v(0));
      Cache(0, grid_v(1));
      Cache(0, grid_v(2));
    }

    // Declare(p);
    For((i, j, k, p_ptr), l, [&] {
      auto p = Eval(l[i, j, k, p_ptr]);
      // For(p, particle_x(0), [&] {
      auto x = particle_x[p];
      auto v = Vector(dim);
      Mutable(v, DataType::f32);
      auto C = Matrix(dim, dim);
      Mutable(C, DataType::f32);

      for (int i = 0; i < dim; i++) {
        v(i) = Expr(0.0_f);
        for (int j = 0; j < dim; j++) {
          C(i, j) = Expr(0.0_f);
        }
      }

      auto base_coord = floor(inv_dx * x - 0.5_f);
      auto fx = x * inv_dx - base_coord;

      Vector w[] = {Eval(0.5_f * sqr(1.5_f - fx)),
                    Eval(0.75_f - sqr(fx - 1.0_f)),
                    Eval(0.5_f * sqr(fx - 0.5_f))};

      int low = 0, high = 1;
      auto base_coord_i =
          Eval(AssumeInRange(cast<int32>(base_coord(0)), i, low, high));
      auto base_coord_j =
          Eval(AssumeInRange(cast<int32>(base_coord(1)), j, low, high));
      auto base_coord_k =
          Eval(AssumeInRange(cast<int32>(base_coord(2)), k, low, high));

      /*
      auto base_coord_i = cast<int32>(base_coord(0));
      auto base_coord_j = cast<int32>(base_coord(1));
      auto base_coord_k = cast<int32>(base_coord(2));
      */

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
      particle_v(1).val<float32>(i) = -13.0_f;
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

  Vector2i cam_res(1280, 720);

  // int scale = 128 * 6 / n;
  // GUI gui("MPM", n * scale + 200, n * scale);
  GUI gui("MPM", cam_res);
  // int angle = 0;
  // int gravity_x_slider = 0;
  /*
  gui.button("Restart", reset)
      .slider("Camera", angle, 0, 360, 1)
      .slider("Gravity Dir", gravity_x_slider, -100, 100);
      */

  auto renderer = create_instance_unique<ParticleRenderer>("shadow_map");
  auto radius = 1.0_f;

  auto simulate_frame = [&]() {
    for (int t = 0; t < 60; t++) {
      TC_PROFILE("reset grid", reset_grid());
      TC_PROFILE("p2g", p2g());
      TC_PROFILE("grid_op", grid_op());
      TC_PROFILE("g2p", g2p());
    }
  };

  Dict cam_dict;
  cam_dict.set("origin", Vector3(radius * 0.6f, radius * 0.3_f, radius * 0.6_f))
      .set("look_at", Vector3(0, -0.2f, 0))
      .set("up", Vector3(0, 1, 0))
      .set("fov", 70)
      .set("res", cam_res);
  auto cam = create_instance<Camera>("pinhole", cam_dict);

  Dict dict;
  dict.set("shadow_map_resolution", 0.005_f)
      .set("alpha", 0.5_f)
      .set("shadowing", 0.008_f)
      .set("ambient_light", 0.5_f)
      .set("light_direction", Vector3(1, 0.5, 0.3));

  renderer->initialize(dict);
  renderer->set_camera(cam);

  auto &canvas = gui.get_canvas();
  for (int frame = 1;; frame++) {
    simulate_frame();
    auto res = canvas.img.get_res();
    Array2D<Vector3> image(Vector2i(res), Vector3(1) - Vector3(0.0_f));
    /*
    std::vector<Particle> particles;
    read_from_binary_file(particles,
                          fmt::format("{}/{:06d}.tcb", folder, frame));
                          */
    std::vector<RenderParticle> render_particles;
    for (int i = 0; i < n_particles; i++) {
      auto x = particle_x(0).val<float32>(i), y = particle_x(1).val<float32>(i),
           z = particle_x(2).val<float32>(i);
      // auto color = hsv2rgb(Vector3(fract(p.pos[3] / 4) * 2, 0.7_f, 0.9_f));
      auto pos = Vector3(x, y, z);
      pos = pos - Vector3(0.5f);
      pos = pos * Vector3(0.5f);
      render_particles.push_back(
          RenderParticle(pos, Vector4(0.6f, 0.7f, 0.9f, 1.0_f)));
    }

    renderer->render(image, render_particles);
    for (auto &ind : image.get_region()) {
      canvas.img[ind] = Vector4(image[ind]);
    }
    gui.update();
    auto render_dir = fmt::format("{}_rendered", "mpm");
    create_directories(render_dir);
    gui.get_canvas().img.write_as_image(
        fmt::format("{}/{:05d}.png", render_dir, frame));
  }

  /*
  int frame = 0;
  for (int f = 0;; f++) {
    TC_PROFILER("mpm 3d");
    simulate_frame();
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
  */
};
TC_REGISTER_TASK(mpm3d);

TC_NAMESPACE_END
