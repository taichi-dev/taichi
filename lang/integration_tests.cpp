#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <taichi/visual/gui.h>

TLANG_NAMESPACE_BEGIN

static TC_FORCE_INLINE __m128 make_float4(float32 a,
                                          float32 b,
                                          float32 c,
                                          float32 d) {
  return _mm_set_ps(d, c, b, a);
}

TC_FORCE_INLINE __m128 make_float3(float a, float b, float c) {
  return make_float4(a, b, c, 0);
}

// clang-format off
TC_ALIGNED(64) const static __m128 grid_pos_offset_[27] = {
    make_float3(0, 0, 0),
    make_float3(0, 0, 1),
    make_float3(0, 0, 2),
    make_float3(0, 1, 0),
    make_float3(0, 1, 1),
    make_float3(0, 1, 2),
    make_float3(0, 2, 0),
    make_float3(0, 2, 1),
    make_float3(0, 2, 2),
    make_float3(1, 0, 0),
    make_float3(1, 0, 1),
    make_float3(1, 0, 2),
    make_float3(1, 1, 0),
    make_float3(1, 1, 1),
    make_float3(1, 1, 2),
    make_float3(1, 2, 0),
    make_float3(1, 2, 1),
    make_float3(1, 2, 2),
    make_float3(2, 0, 0),
    make_float3(2, 0, 1),
    make_float3(2, 0, 2),
    make_float3(2, 1, 0),
    make_float3(2, 1, 1),
    make_float3(2, 1, 2),
    make_float3(2, 2, 0),
    make_float3(2, 2, 1),
    make_float3(2, 2, 2),
};
// clang-format on

struct MPMContext {
 public:
  struct Particle {
    Vector3 pos, v;
    Matrix3 C;
    real J;

    Particle() {
      pos = Vector3::rand() * 0.5_f + Vector3(0.25_f);
      v = Vector3::rand();
      // C = Matrix3::rand();
      J = 1.0_f * rand() + 1;
    }
  };

  int n_particles;  // num particles
  static constexpr real mass = 1.0_f;
  static constexpr real vol = 1.0_f;
  static constexpr int n = 8;
  static constexpr real dx = 1.0_f / n;
  static constexpr real dt = 1e-3_f;

  using Grid = Vector4[n][n][n];
  Grid grid;

  void clear_grid() {
    std::memset(grid, 0, sizeof(grid));
  }

  std::vector<Particle> particles;

  MPMContext(int n_particles) : n_particles(n_particles) {
    particles.resize(n_particles);
    for (int i = 0; i < n_particles; i++)
      particles[i] = Particle();
    clear_grid();
  }

  void p2g() {
    const auto inv_dx = 1.0_f / dx;
    for (int p_i = 0; p_i < n_particles; p_i++) {
      using Vec = Vector3;
      auto &p = particles[p_i];
      Vector3i base_coord = (p.pos * inv_dx - Vector3(0.5_f)).cast<int>();
      auto fx = p.pos * inv_dx - base_coord.cast<real>();
      // Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx,
      // fx-1,fx-2]
      Vec w[3]{Vec(0.5) * sqr(Vec(1.5) - fx), Vec(0.75) - sqr(fx - Vec(1.0)),
               Vec(0.5) * sqr(fx - Vec(0.5))};
      real J = p.J;
      auto stress =  // Cauchy stress times dt and inv_dx
          -4 * inv_dx * inv_dx * dt * vol * Matrix3(J - 1);
      auto affine = stress + mass * p.C;

      // Scatter to grid
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            auto dpos = (Vec(i, j, k) - fx) * dx;
            Vector4 mv(p.v * mass, mass);  // translational momentum
            grid[base_coord.x + i][base_coord.y + j][base_coord.z + k] +=
                w[i].x * w[j].y * w[k].z * (mv + Vector4(affine * dpos, 0));
          }
        }
      }
    }
  }

  using Vector = Vector3f;
  using Matrix = Matrix3f;
  using Vectori = Vector3i;

  struct MLSMPMFastKernel32 {
    static constexpr int dim = 3;
    TC_ALIGNED(64) __m128 kernels[3][3];

    TC_FORCE_INLINE MLSMPMFastKernel32(const __m128 &pos, real inv_delta_x) {
      __m128 p_fract = _mm_sub_ps(pos, _mm_set1_ps(0.5f));
      TC_ALIGNED(64) __m128 w_cache[dim];
      for (int k = 0; k < dim; k++) {
        __m128 t = _mm_sub_ps(_mm_set1_ps(p_fract[k]),
                              make_float4(-0.5f, 0.5f, 1.5f, 0.0f));
        __m128 tt = _mm_mul_ps(t, t);
        w_cache[k] = _mm_fmadd_ps(
            make_float4(0.5f, -1.0f, 0.5f, 0.0f), tt,
            _mm_fmadd_ps(make_float4(-1.5f, 0.0f, 1.5f, 0.0f), t,
                         make_float4(1.125f, 0.75f, 1.125f, 0.0f)));
      }
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          kernels[i][j] = _mm_mul_ps(_mm_set1_ps(w_cache[0][i] * w_cache[1][j]),
                                     w_cache[2]);
        }
      }
    }
  };

  TC_FORCE_INLINE static int get_stencil_start(real x) {
    return int(x - 0.5f);
  }

#define broadcast(s, i) _mm_shuffle_ps((s), (s), 0x55 * (i))
  void p2g_intrinsics() {
    const auto inv_delta_x = 1.0_f / dx;
    const real delta_t = dt;
    __m128 S = _mm_set1_ps(-4.0_f * inv_delta_x * delta_t);
    for (int p_i = 0; p_i < n_particles; p_i++) {
      Particle &p = particles[p_i];

      // Note, pos is magnified grid pos
      __m128 pos_ = _mm_mul_ps(p.pos.v, _mm_set1_ps(inv_delta_x));
      Vectori grid_base_pos(get_stencil_start(pos_[0]),
                            get_stencil_start(pos_[1]),
                            get_stencil_start(pos_[2]));
      auto base_offset =
          grid_base_pos[0] * n * n + grid_base_pos[1] * n + grid_base_pos[2];
      Vector grid_base_pos_f = Vector(grid_base_pos);

      MLSMPMFastKernel32 kernel(_mm_sub_ps(pos_, grid_base_pos_f), inv_delta_x);
      const __m128(&kernels)[3][3] = kernel.kernels;
      using KernelLinearized = real[3 * 3 * 4];
      const KernelLinearized &kernels_linearized =
          *reinterpret_cast<const KernelLinearized *>(&kernels[0][0][0]);
      const __m128 v = p.v.v;
      __m128 mass_ = _mm_set1_ps(mass);
      // Note, apic_b has delta_x issue
      const Matrix3f apic_b_inv_d_mass = -mass * dx * p.C;
      const __m128 mass_v = _mm_mul_ps(_mm_set1_ps(mass), v);

      auto stress = dt * vol * Matrix3(p.J - 1);

      __m128 delta_t_tmp_force_[3];
      Matrix3 &delta_t_tmp_force =
          reinterpret_cast<Matrix &>(delta_t_tmp_force_);
      for (int i = 0; i < 3; i++) {
        delta_t_tmp_force_[i] = _mm_mul_ps(_mm_set1_ps(delta_t), stress[i]);
      }

      __m128 rela_pos = _mm_sub_ps(pos_, grid_base_pos_f);
      __m128 affine[3];

      for (int i = 0; i < 3; i++)
        affine[i] = _mm_fmadd_ps(stress[i], S, apic_b_inv_d_mass[i]);
#define LOOP(node_id)                                                       \
  {                                                                         \
    __m128 dpos = _mm_sub_ps(rela_pos, grid_pos_offset_[node_id]);          \
    __m128 weight =                                                         \
        _mm_set1_ps(kernels[node_id / 9][node_id / 3 % 3][node_id % 3]);    \
    __m128 affine_prod = _mm_fmadd_ps(                                      \
        affine[2], broadcast(dpos, 2),                                      \
        _mm_fmadd_ps(affine[1], broadcast(dpos, 1),                         \
                     _mm_fmadd_ps(affine[0], broadcast(dpos, 0), mass_v))); \
    __m128 contrib = _mm_blend_ps(mass_, affine_prod, 0x7);                 \
    __m128 delta = _mm_mul_ps(weight, contrib);                             \
    auto &g = grid[0][0][node_id / 9 * n * n + node_id / 3 % 3 * n +        \
                         node_id % 3 + base_offset];                        \
    g = _mm_add_ps(g, delta);                                               \
  }
      TC_REPEAT27(LOOP);
#undef LOOP
    }
  }
#undef broadcast
};

TC_TEST("simd_mpm") {
  while (1) {
    MPMContext context(1);
    context.p2g();

    MPMContext::Grid grid_gt = context.grid;
    context.clear_grid();
    context.p2g_intrinsics();

    for (int i = 0; i < context.n; i++) {
      for (int j = 0; j < context.n; j++) {
        for (int k = 0; k < context.n; k++) {
          for (int d = 0; d < 4; d++) {
            // TC_INFO("{} {} {} {} , {}", i, j, k, d, grid_gt[i][j][k][d]);
            TC_CHECK_EQUAL(grid_gt[i][j][k][d], context.grid[i][j][k][d], 1e-3_f);
          }
        }
      }
    }
  }

  /*
  CoreState::set_trigger_gdb_when_crash(true);
  Program prog(Arch::x86_64);
  prog.config.print_ir = true;

  global(a, i32);
  auto i = Index(0);

  layout([&]() { root.fixed(i, 128).place(a); });

  auto func = kernel([&]() {
    Matrix A(2, 2), B(2, 2);
    A(0, 0) = 1;
    A(0, 1) = 1;
    A(1, 0) = 1;
    A(1, 1) = 1;

    B(0, 0) = 1;
    B(0, 1) = 2;
    B(1, 0) = 3;
    B(1, 1) = 4;
    auto C = A * B + A;
    for (int p = 0; p < 2; p++) {
      for (int q = 0; q < 2; q++) {
        Print(C(p, q));
      }
    }
  });

  func();
  */
};

TLANG_NAMESPACE_END
