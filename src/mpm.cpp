#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>
#include <taichi/util.h>
#include <taichi/math/svd.h>

TC_NAMESPACE_BEGIN

auto theta_c = 2.5e-2_f;
auto theta_s = 7.5e-3_f;
auto E = 1.4e4_f;
// auto E = 0;
auto nu = 0.2_f;
auto lambda_0 = E * nu / ((1.0_f + nu) * (1.0_f - 2.0_f * nu));
auto mu_0 = E / (2.0_f * (1.0_f + nu));
auto min_Jp = 0.6_f;
auto max_Jp = 20.0_f;
auto hardening = 0.0_f;

class Particle {
  static constexpr int dim = 3;

 public:
  using Vector = TVector<real, dim>;
  using VectorP = TVector<real, dim + 1>;
  using Matrix = TMatrix<real, dim>;

  Matrix dg_e;
  // Affine momemtum (APIC)
  Matrix apic_b;
  Vector pos;
  // Elastic deformation gradient
  real Jp;  // determinant of dg_p = determinant(plastic deformation gradient)

 private:
  VectorP v_and_m;

 public:
  Particle() {
  }

  Particle(Vector pos, Vector v) : pos(pos) {
    dg_e = Matrix(1);
    apic_b = Matrix(0);
    Jp = 1.0_f;
    v_and_m = VectorP(v, 1);
  }

  TC_FORCE_INLINE real get_mass() const {
    return v_and_m[dim];
  }

  TC_FORCE_INLINE void set_mass(real mass) {
    v_and_m[dim] = mass;
  }

  TC_FORCE_INLINE Vector get_velocity() const {
    return Vector(v_and_m);
  }

  TC_FORCE_INLINE void set_velocity(const Vector &v) {
    v_and_m = VectorP(v, v_and_m[dim]);
  }

  TC_FORCE_INLINE void set_velocity(const __m128 &v) {
    v_and_m = _mm_blend_ps(v, v_and_m, 0x7);
  }

  Matrix first_piola_kirchhoff() {
    real j_e = determinant(this->dg_e);
    auto lame = get_lame_parameters();
    real mu = lame.first, lambda = lame.second;
    Matrix r, s;
    polar_decomp(this->dg_e, r, s);
    Matrix grad = 2 * mu * (this->dg_e - r) +
                  lambda * (j_e - 1) * j_e * inverse(transpose(this->dg_e));
    return grad;
  }

  Matrix calculate_force() {
    // 2 = mass / rho; rho = 0.5
    return -2.0_f * first_piola_kirchhoff() * transpose(this->dg_e);
  }

  void plasticity(const Matrix &cdg) {
    this->dg_e = cdg * this->dg_e;
    real dg_e_det_orig = 1.0_f;
    real dg_e_det = 1.0_f;
    Matrix svd_u, sig, svd_v;
    svd(this->dg_e, svd_u, sig, svd_v);
    for (int i = 0; i < dim; i++) {
      dg_e_det_orig *= sig[i][i];
      sig[i][i] = clamp(sig[i][i], 1.0_f - theta_c, 1.0_f + theta_s);
      dg_e_det *= sig[i][i];
    }
    this->dg_e = svd_u * sig * transposed(svd_v);

    real Jp_new = Jp * dg_e_det_orig / dg_e_det;
    if (!(Jp_new <= max_Jp))
      Jp_new = max_Jp;
    if (!(Jp_new >= min_Jp))
      Jp_new = min_Jp;
    Jp = Jp_new;
  }

  std::pair<real, real> get_lame_parameters() const {
    real e = std::exp(hardening * (1.0_f - Jp));
    real mu = mu_0 * e;
    real lambda = lambda_0 * e;
    return {mu, lambda};
  }
};

using Block = TBlock<Vector4f, Particle, TSize3D<8>, 1>;

template <int dim, int order_>
struct MPMKernelBase {
  constexpr static int D = dim;
  constexpr static int order = order_;
  constexpr static int kernel_size = order + 1;

  using Vector = VectorND<dim, real>;
  using VectorP = VectorND<dim + 1, real>;
  using VectorI = VectorND<dim, int>;

  VectorP w_stages[D][kernel_size];
  Vector4 w_cache[D];
  Vector4 dw_cache[D];
  VectorP kernels;
  real inv_delta_x;

  TC_FORCE_INLINE void shuffle() {
    for (int k = 0; k < kernel_size; k++) {
      for (int j = 0; j < D; j++) {
        w_stages[j][k] = VectorP([&](int i) -> real {
          if (j == i) {
            return dw_cache[j][k] * inv_delta_x;
          } else {
            return w_cache[j][k];
          }
        });
      }
    }
  }

  TC_FORCE_INLINE VectorP get_dw_w(const VectorI &k) const {
    VectorP ret = w_stages[0][k[0]];
    for (int i = 1; i < dim; i++) {
      ret *= w_stages[i][k[i]];
    }
    return ret;
  }

  TC_FORCE_INLINE Vector get_dw(const VectorI &k) const {
    VectorP ret = w_stages[0][k[0]];
    for (int i = 1; i < dim; i++) {
      ret *= w_stages[i][k[i]];
    }
    return Vector(ret);
  }

  TC_FORCE_INLINE real get_w(const VectorI &k) const {
    VectorP ret = w_stages[0][k[0]];
    for (int i = 1; i < dim; i++) {
      ret *= w_stages[i][k[i]];
    }
    return ret[D];
  }

  static TC_FORCE_INLINE constexpr real inv_D() {
    return 6.0f - real(order);
  }
};

template <int dim, int order>
struct MPMKernel;

// Quadratic kernel
template <int dim>
struct MPMKernel<dim, 2> : public MPMKernelBase<dim, 2> {
  using Base = MPMKernelBase<dim, 2>;
  using Vector = typename Base::Vector;

  TC_FORCE_INLINE MPMKernel(const Vector &pos,
                            real inv_delta_x,
                            bool do_calculate_kernel = true,
                            bool do_shuffle = true) {
    if (do_calculate_kernel)
      calculate_kernel(pos);
    this->inv_delta_x = inv_delta_x;
    if (do_shuffle)
      this->shuffle();
  }

  // NOTE: x may be negative in pangu!
  TC_FORCE_INLINE static int get_stencil_start(real x) {
    return int(std::floor(x - 0.5f));
  }

  TC_FORCE_INLINE void calculate_kernel(const Vector &pos) {
    Vector p_fract = fract(pos - Vector(0.5f));
    for (int k = 0; k < dim; k++) {
      const Vector4 t = Vector4(p_fract[k]) - Vector4(-0.5f, 0.5f, 1.5f, 0.0f);
      auto tt = t * t;
      this->w_cache[k] = Vector4(0.5f, -1.0f, 0.5f, 0.0f) * tt +
                         Vector4(-1.5f, 0.0f, 1.5f, 0.0f) * t +
                         Vector4(1.125f, 0.75f, 1.125f, 0.0f);
      this->dw_cache[k] =
          Vector4(1.0f, -2.0f, 1.0f, 0.0f) * t + Vector4(-1.5f, 0, 1.5f, 0.0f);
    }
  }
};

class MPMTest {
 public:
  static constexpr auto dim = Block::dim;
  using Vector = TVector<real, dim>;
  using VectorP = TVector<real, dim + 1>;
  using Matrix = TMatrix<real, dim>;
  using Grid = TaichiGrid<Block>;

  Grid grid;

  real dx, inv_dx, dt;  // Cell size and dt
  real frame_dt;
  int total_frames;
  using VectorI = Vector3i;
  using Vectori = VectorI;
  int current_frame;
  Vector3 gravity;
  using Kernel = MPMKernel<3, 2>;
  Block::particle_to_grid_func grid_pos;

  MPMTest() {
    gravity = Vector3(0, -100, 0);
    current_frame = 0;
    auto res = 30;
    dx = 1.0_f / res;
    dt = 5e-5_f;
    frame_dt = 1e-2f;
    //frame_dt = 5e-5_f;
    inv_dx = 1.0_f / dx;
    total_frames = 128;
    grid_pos = [&](Particle &p) -> VectorI {
      return (p.pos * inv_dx + Vector(0.5_f)).floor().template cast<int>();
    };
    Region3D reg(VectorI(-res / 2, res, -res / 2),
                 VectorI(res / 2, res + res / 2, res / 2));
    for (auto &ind : reg) {
      for (int i = 0; i < 8; i++) {
        auto pos = (ind.get_pos() + Vector3::rand()) * dx;
        pos.y += pos.x;
        Particle p(pos, Vector(0, -10, 0));
        auto node = grid_pos(p);
        if (grid.inside(node)) {
          grid.touch(node);
          auto b = grid.get_block_if_exist(node);
          b->add_particle(p);
        }
      }
    }
    output(get_filename(current_frame));
  }

  TC_FORCE_INLINE Vectori get_grid_base_pos(const Vector &pos) const {
    return Vectori(
        [&](int i) -> int { return Kernel::get_stencil_start(pos[i]); });
  }

  virtual void substep() {
    auto delta_x = dx;
    auto inv_delta_x = inv_dx;
    auto delta_t = dt;
    {
      TC_PROFILER("P2G");
      // P2G
      grid.advance(
          [&](Grid::Block &b, Grid::Ancestors &an) {
            gather_particles(b, an, grid_pos);

            // Rasterize
            for (std::size_t i = 0; i < b.particle_count; i++) {
              auto &p = b.particles[i];

              const Vector pos = p.pos * inv_dx;
              const Vector v = p.get_velocity();
              const real mass = p.get_mass();
              // Note, apic_b has delta_x issue
              const Matrix apic_b_inv_d_mass = p.apic_b * (4 * mass);
              const Vector mass_v = mass * v;
              Matrix delta_t_tmp_force = dt * p.calculate_force();
              RegionND<dim> region(VectorI(0), VectorI(Kernel::kernel_size));

              VectorI grid_base_pos = get_grid_base_pos(pos);
              Kernel kernel(pos, inv_dx);

              for (auto &ind : region) {
                auto i = ind.get_ipos() + grid_base_pos;
                Vector dpos = pos - i.template cast<real>();
                VectorP dw_w = kernel.get_dw_w(ind.get_ipos());

                VectorP delta =
                    dw_w[dim] *
                    (VectorP(mass_v + apic_b_inv_d_mass * dpos, mass) +
                     VectorP(-delta_t_tmp_force * dpos * 4.0_f * inv_dx));
                b.node_global(i) += delta;
              }
            }
            return b.particle_count != 0;
          },
          true);
    }
    {
      TC_PROFILER("G2P");
      grid.advance(
          [&](Grid::Block &b, Grid::Ancestors &an) {
            gather_particles(b, an, grid_pos);
            accumulate_dilated_grids(b, an);
            // normalize grid
            if (b.base_coord.y <= 0) {
              for (auto &g : b.nodes) {
                if (g[dim] > 0) {
                  g *= 1.0_f / g[dim];
                  g.y = std::max(g.y, 0.0_f);
                  g[dim] = 0;
                }
              }
            } else {
              for (auto &g : b.nodes) {
                if (g[dim] > 0) {
                  g *= 1.0_f / g[dim];
                  g += VectorP(gravity * dt);
                  g[dim] = 0;
                }
              }
            }

            // Resample
            for (std::size_t t = 0; t < b.particle_count; t++) {
              auto &p = b.particles[t];
              Vector v(0.0f);
              Matrix B(0.0f);
              Matrix cdg(0.0f);
              Vector pos = p.pos * inv_delta_x;

              RegionND<dim> region(VectorI(0), VectorI(Kernel::kernel_size));
              Vectori grid_base_pos = get_grid_base_pos(pos);
              Kernel kernel(pos, inv_delta_x);

              for (auto &ind : region) {
                auto i = ind.get_ipos() + grid_base_pos;
                auto grid_vel = Vector3(b.node_global(i));
                Vector dpos = pos - i.template cast<real>();
                VectorP dw_w = kernel.get_dw_w(ind.get_ipos());
                v += dw_w[dim] * grid_vel;
                B += Matrix::outer_product(dw_w[dim] * grid_vel, dpos);
              }
              p.apic_b = B;
              p.set_velocity(v);
              // grad v
              cdg = B * (-4 * inv_delta_x);
              cdg = Matrix(1.0f) + delta_t * cdg;
              // Plasticity
              p.plasticity(cdg);
              // Advection
              p.pos += delta_t * v;
            }
            return true;
          },
          false);
    }
  }

  void advance() {
    TC_P(current_frame);
    {
      // TC_PROFILER("simulation");
      for (int i = 0; i < int(std::round(frame_dt / dt)); i++) {
        TC_PROFILE("substep", substep());
      }
    }
    current_frame += 1;
    // TC_PROFILE("output", output(get_filename(current_frame)));
    output(get_filename(current_frame));
    TC_P(grid.num_active_blocks());
    TC_P(grid.num_particles());
  }

  std::string get_filename(int frame) {
    return fmt::format("/tmp/outputs/{:05d}.tcb", current_frame);
  }

  virtual void output(std::string fn) {
    if (!grid.is_master()) {
      return;
    }
    OptiXScene scene;
    grid.serial_for_each_particle([&](Particle &p) {
      scene.particles.push_back(OptiXParticle{Vector4(p.pos * 3.0_f, 0.01_f)});
    });
    write_to_binary_file(scene, fn);
  }
};

auto mpm = [](const std::vector<std::string> &params) {
  // ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MPMTest> mpm;
  mpm = std::make_unique<MPMTest>();
  for (int t = 0; t < mpm->total_frames; t++) {
    TC_PROFILE("advance", mpm->advance());
    taichi::print_profile_info();
  }
};

TC_REGISTER_TASK(mpm);

TC_NAMESPACE_END
