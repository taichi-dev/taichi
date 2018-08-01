#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>
#include <taichi/util.h>
#include <taichi/math/svd.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/gui.h>

TC_NAMESPACE_BEGIN

real gravity = 100;
real buoyancy = 700;
real temperature_decay = 1;

static constexpr bool debug = false;

struct BlockFlags : public bit::Bits<32> {
  using Base = bit::Bits<32>;
  // TC_BIT_FIELD(uint8, num_effective_neighbours, 0);
  TC_BIT_FIELD(bool, has_effective_cell, 8);
};

struct NodeFlags : public bit::Bits<32> {
  using Base = bit::Bits<32>;
  TC_BIT_FIELD(uint8, num_effective_neighbours, 0);
  TC_BIT_FIELD(bool, effective, 8);
};

struct Node {
  real channels[16];

  real &operator[](int i) {
    return channels[i];
  }

  NodeFlags &flags() {
    return bit::reinterpret_bits<NodeFlags>(channels[15]);
  }
};

struct Particle {
  Vector3 pos;
};

Vector3 hsv2rgb(Vector3 hsv) {
  real h = hsv.x;
  real s = hsv.y;
  real v = hsv.z;
  int j = (int)floor(h * 6);
  real f = h * 6 - j;
  real p = v * (1 - s);
  real q = v * (1 - f * s);
  real t = v * (1 - (1 - f) * s);
  real r, g, b;
  switch (j % 6) {
    case 0:
      r = v, g = t, b = p;
      break;
    case 1:
      r = q, g = v, b = p;
      break;
    case 2:
      r = p, g = v, b = t;
      break;
    case 3:
      r = p, g = q, b = v;
      break;
    case 4:
      r = t, g = p, b = v;
      break;
    default:  // 5, actually
      r = v, g = p, b = q;
      break;
  }
  return Vector3(r, g, b);
}

using Block = TBlock<Node, Particle, TSize3D<8>, 0, 1, BlockFlags>;

class MGPCGSmoke {
 public:
  static constexpr auto dim = Block::dim;
  using Vector = TVector<real, dim>;
  using VectorP = TVector<real, dim + 1>;
  using Matrix = TMatrix<real, dim>;
  using Grid = TaichiGrid<Block>;

  const int n = 64;
  static constexpr int mg_lv = 4;
  std::vector<std::unique_ptr<Grid>> grids;

  enum {
    CH_R,
    CH_Z,
    CH_X,
    CH_B,
    CH_P,
    CH_MG_U,
    CH_MG_B,
    CH_MG_R,
    CH_VX,
    CH_VY,
    CH_VZ,
    CH_RHO,
    CH_T
  };

  using VectorI = Vector3i;
  using Vectori = VectorI;
  using GridScratchPad = TGridScratchPad<Block>;
  using GridScratchPadCh = TGridScratchPad<Block, real>;

  std::shared_ptr<Camera> cam;
  real current_t;
  real dt = 2e-3_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;

  std::unique_ptr<ParticleRenderer> renderer;

  MGPCGSmoke() {
    renderer = create_instance_unique<ParticleRenderer>("shadow_map");
    auto radius = 1.0_f;

    Dict cam_dict;
    cam_dict.set("origin", Vector(0, radius * 0.3, radius))
        .set("look_at", Vector(0, 0, 0))
        .set("up", Vector(0, 1, 0))
        .set("fov", 70)
        .set("res", Vector2i(800));
    cam = create_instance<Camera>("pinhole", cam_dict);

    Dict dict;
    dict.set("shadow_map_resolution", 0.005_f)
        .set("alpha", 0.6_f)
        .set("shadowing", 0.01_f)
        .set("ambient_light", 0.3_f)
        .set("light_direction", Vector(1, 2, 0.5));

    renderer->initialize(dict);
    renderer->set_camera(cam);
    current_t = 0;
    // Span a region in
    grids.resize(mg_lv);
    for (int i = 0; i < mg_lv; i++) {
      grids[i] = std::make_unique<Grid>();
    }
    TC_ASSERT(mg_lv >= 1);
    TC_ASSERT_INFO(bit::is_power_of_two(n), "Only POT grid sizes supported");
    Region3D active_region(VectorI(-n, -n * 2, -n), VectorI(n, n * 2, n));
    for (auto ind : active_region) {
      grids[0]->touch(ind);
      grids[0]->node(ind).flags().set_effective(true);
      grids[0]->get_block_if_exist(ind)->meta.set_has_effective_cell(true);
    }
    set_up_hierechy();
    Region3D buffer_region(VectorI(-n, -n * 2, -n),
                           VectorI(n, n * 2, n) + VectorI(1));
    // Touch extra faces for u, v, w
    for (auto ind : buffer_region) {
      grids[0]->touch(ind);
    }
  }

  void test_pcg() {
    {
      TC_PROFILER("Initialize")
      Region3D active_region(VectorI(-n, -n * 2, -n), VectorI(n, n * 2, n));
      for (auto &ind : active_region) {
        if (ind.get_ipos() == VectorI(0)) {
          grids[0]->node(ind.get_ipos())[CH_B] = 1;
        }
      }
    }
    poisson_solve();
  }

  void set_up_hierechy() {
    std::size_t total_blocks = grids[0]->num_active_blocks();
    for (int i = 0; i < mg_lv - 1; i++) {
      grids[i]->coarsen_to(
          *grids[i + 1], [&](Block &b, Grid::PyramidAncestors &an) {
            for (auto ind : b.local_region()) {
              b.node_local(ind.get_ipos()).flags().set_effective(true);
            }
            b.meta.set_has_effective_cell(true);
          });
      total_blocks /= 8;
      TC_ASSERT(grids[i + 1]->num_active_blocks() == total_blocks);
    }
  }

  void residual(int level, int U, int B, int R) {
    TC_PROFILER("residual");
    grids[level]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          if (!b.meta.get_has_effective_cell())
            return;
          GridScratchPad scratch(an);
          // 6 neighbours
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              for (int k = 0; k < Block::size[2]; k++) {
                auto rhs = b.get_node_volume()[i][j][k][B];
                auto c = b.get_node_volume()[i][j][k][U];
                auto fetch = [&](int ii, int jj, int kk) {
                  rhs += (scratch.data[i + ii][j + jj][k + kk][U] - c);
                };
                fetch(0, 0, 1);
                fetch(0, 0, -1);
                fetch(0, 1, 0);
                fetch(0, -1, 0);
                fetch(1, 0, 0);
                fetch(-1, 0, 0);
                b.get_node_volume()[i][j][k][R] = rhs;
              }
            }
          }
        },
        false, true);
  }

  void multiply(int channel_out, int channel_in) {
    TC_PROFILER("multiply");
    // TODO: this supports zero-Dirichlet BC only!
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          if (!b.meta.get_has_effective_cell())
            return;
          GridScratchPad scratch(an);
          // 6 neighbours
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              for (int k = 0; k < Block::size[2]; k++) {
                int count = 0;
                real tmp = 0;
                auto fetch = [&](int ii, int jj, int kk) {
                  auto &n = scratch.data[i + (ii)][j + (jj)][k + (kk)];
                  count++;
                  tmp += n[channel_in];
                };
                fetch(0, 0, 1);
                fetch(0, 0, -1);
                fetch(0, 1, 0);
                fetch(0, -1, 0);
                fetch(1, 0, 0);
                fetch(-1, 0, 0);
                auto &o = b.get_node_volume()[i][j][k][channel_out];
                o = count * scratch.data[i][j][k][channel_in] - tmp;
                if (debug) {
                  if (o != o) {
                    TC_P(b.base_coord);
                    TC_P(scratch.data[i][j][k][channel_in]);
                    TC_P(o);
                    TC_P(i);
                    TC_P(j);
                    TC_P(k);
                    Time::sleep(0.01);
                  }
                }
              }
            }
          }
        },
        false, true);
  }

  // out += a + scale * b
  void saxpy(int channel_out, int channel_a, int channel_b, real scale) {
    TC_ASSERT(!with_mpi());
    grids[0]->map([&](Block &b) {
      for (auto &n : b.nodes) {
        n[channel_out] = n[channel_a] + scale * n[channel_b];
      }
    });
  }

  // out += a + scale * b
  void copy(int channel_out, int channel_a) {
    TC_PROFILER("copy")
    grids[0]->map([&](Block &b) {
      for (auto &n : b.nodes) {
        n[channel_out] = n[channel_a];
      }
    });
  }

  float64 dot_product(int channel_a, int channel_b) {
    return grids[0]->reduce([&](Block &b) -> float64 {
      float64 sum = 0;
      for (auto &n : b.nodes) {
        sum += n[channel_a] * n[channel_b];
      }
      return sum;
    });
  }

  void smooth(int level, int U, int B) {
    TC_PROFILER("smoothing")
    // TODO: this supports zero-Dirichlet BC only!
    grids[level]->advance(
        [&](Grid::Block &b, Grid::Ancestors &an) {
          if (!b.meta.get_has_effective_cell())
            return;
          GridScratchPadCh scratchB(an, B * sizeof(real));
          GridScratchPadCh scratchU(an, U * sizeof(real));
          // 6 neighbours
          TC_STATIC_ASSERT(sizeof(real) == 4);
          TC_STATIC_ASSERT(Block::size[2] == 8);
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              __m256 sum_z =
                  _mm256_add_ps(_mm256_loadu_ps(&scratchU.data[i][j][-1]),
                                _mm256_loadu_ps(&scratchU.data[i][j][1]));
              __m256 sum_y =
                  _mm256_add_ps(_mm256_loadu_ps(&scratchU.data[i][j - 1][0]),
                                _mm256_loadu_ps(&scratchU.data[i][j + 1][0]));
              __m256 sum_x =
                  _mm256_add_ps(_mm256_loadu_ps(&scratchU.data[i - 1][j][0]),
                                _mm256_loadu_ps(&scratchU.data[i + 1][j][0]));

              auto sum = _mm256_add_ps(
                  _mm256_add_ps(sum_z, sum_y),
                  _mm256_add_ps(sum_x,
                                _mm256_loadu_ps(&scratchB.data[i][j][0])));
              auto original = _mm256_loadu_ps(&scratchU.data[i][j][0]);

              // o = original + (tmp * (1.0_f / 6) - original) * (2.0_f / 3_f);
              sum = _mm256_add_ps(
                  original,
                  _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(sum, _mm256_set1_ps(
                                                                     1.0f / 6)),
                                              original),
                                _mm256_set1_ps(2.0f / 3)));
              real *psum = (real *)&sum;
              for (int k = 0; k < Block::size[2]; k++) {
                // (B - Lu) / Diag
                b.get_node_volume()[i][j][k][U] = psum[k];
                // Damping is important. It brings down #iterations to 1e-7 from
                // 91 to 10...
              }
            }
          }
        },
        false, true);  // carry nodes only if on finest level
  }

  void clear(int level, int channel) {
    grids[level]->for_each_node([&](Block::Node &n) { n[channel] = 0; });
  }

  // B[level + 1] = coarsened(R[level])
  void restrict(int level, int R_in, int B_out) {
    TC_PROFILER("restriction");
    // NOTE: supports POT grids only
    // sum residual
    clear(level + 1, B_out);
    grids[level]->coarsen_to(
        *grids[level + 1], [&](Block &block, Grid::PyramidAncestors &an) {
          if (!block.meta.get_has_effective_cell())
            return;
          for (auto ind : Region3D(Vector3i(0), Vector3i(2))) {
            if (!an[ind.get_ipos()]) {
              TC_NOT_IMPLEMENTED
              continue;
            }
            Block &ab = *an[ind.get_ipos()];
            for (auto j : ab.local_region()) {
              auto coarse_coord = div_floor(
                  ind.get_ipos() * Vector3i(Block::size) + j.get_ipos(),
                  Vector3i(2));
              block.node_local(coarse_coord).channels[B_out] +=
                  ab.node_local(j.get_ipos())[R_in];
            }
          }
        });
  }

  // U[level] += refined(U]level + 1]);
  void prolongate(int level, int U) {
    TC_PROFILER("prolongation");
    real scale = 0.5_f;
    // upsample and apply correction
    grids[level]->refine_from(
        *grids[level + 1], [&](Block &block, Block &ancestor) {
          if (!block.meta.get_has_effective_cell())
            return;
          for (auto ind : block.global_region()) {
            auto correction =
                scale *
                ancestor.node_global(div_floor(ind.get_ipos(), Vector3i(2)))[U];
            block.node_global(ind.get_ipos())[U] += correction;
          }
        });
  }

  real norm(int channel) {
    return (real)std::sqrt(dot_product(channel, channel));
  }

  void V_cycle(int channel_in,
               int channel_out,
               bool use_as_preconditioner = true) {
    copy(CH_MG_B, channel_in);
    constexpr int U = CH_MG_U, B = CH_MG_B, R = CH_MG_R;
    constexpr int smoothing_iters = 2, bottom_smoothing_iter = 150;
    if (use_as_preconditioner) {
      clear(0, U);
    }
    for (int i = 0; i < mg_lv - 1; i++) {
      // pre-smoothing
      for (int j = 0; j < smoothing_iters * 1000000; j++) {
        TC_TIME(smooth(i, U, B));
      }
      residual(i, U, B, R);
      restrict(i, R, B);
      clear(i + 1, U);
    }

    {
      TC_PROFILER("bottom smoothing");
      for (int j = 0; j < bottom_smoothing_iter; j++) {
        smooth(mg_lv - 1, U, B);
      }
    }

    for (int i = mg_lv - 2; i >= 0; i--) {
      prolongate(i, U);
      // post-smoothing
      for (int j = 0; j < smoothing_iters; j++) {
        smooth(i, U, B);
      }
    }
    copy(channel_out, CH_MG_U);
  }

  void run() {
    for (int i = 0; i < 1000000; i++) {
      V_cycle(CH_B, CH_X, false);
      residual(0, CH_X, CH_B, CH_R);
      auto residual_l2 = norm(CH_R);
      TC_TRACE("iter {}, residual {}", i, residual_l2);
      if (residual_l2 < 1e-7) {
        break;
      }
    }
  }

  // https://en.wikipedia.org/wiki/Conjugate_gradient_method
  void poisson_solve() {
    TC_PROFILER("Poisson Solve");
    constexpr real tolerance = 1e-4_f;
    bool use_preconditioner = true;
    real initial_residual_norm = norm(CH_B);
    TC_P(initial_residual_norm);
    if (initial_residual_norm < 1e-20_f) {
      return;
    }
    clear(0, CH_X);
    // r = b - Ax
    residual(0, CH_X, CH_B, CH_R);
    // z = M^-1 r
    if (use_preconditioner) {
      V_cycle(CH_R, CH_Z);
    } else {
      copy(CH_Z, CH_R);
    }
    // p = z
    copy(CH_P, CH_Z);
    auto old_zr = dot_product(CH_Z, CH_R);
    for (int i = 0; i < 100000; i++) {
      TC_PROFILE("Multiply", multiply(CH_Z, CH_P));
      real alpha = old_zr / dot_product(CH_P, CH_Z);

      saxpy(CH_X, CH_X, CH_P, alpha);
      saxpy(CH_R, CH_R, CH_Z, -alpha);

      auto residual_l2 = norm(CH_R);
      TC_TRACE("iter {}, residual {}", i, residual_l2);
      if (residual_l2 < tolerance * initial_residual_norm) {
        break;
      }

      if (use_preconditioner) {
        TC_PROFILE("V_cycle", V_cycle(CH_R, CH_Z));
      } else {
        copy(CH_Z, CH_R);
      }

      auto new_zr = dot_product(CH_Z, CH_R);
      auto beta = new_zr / old_zr;
      old_zr = new_zr;
      saxpy(CH_P, CH_Z, CH_P, beta);
    }
  }

  TC_FORCE_INLINE Vector3 storage_offset(int axis) {
    return axis < 3 ? (Vector3(1) - Vector3::axis(axis)) * 0.5_f
                    : Vector3(0.5_f);
  }

  void advect() {
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          TC_STATIC_ASSERT(Block::size[0] == 8);
          auto scale = Vector(inv_dx);
          auto corner = b.base_coord.template cast<real>() -
                        VectorI(Block::size).template cast<real>();

          // Unfortunately, ux, uy, uz and density are not collocated...
          LerpField<real, TSize3D<24>> u(scale, corner + storage_offset(0));
          LerpField<real, TSize3D<24>> v(scale, corner + storage_offset(1));
          LerpField<real, TSize3D<24>> w(scale, corner + storage_offset(2));
          LerpField<real, TSize3D<24>> rho(scale, corner + storage_offset(3));
          LerpField<real, TSize3D<24>> T(scale, corner + storage_offset(3));

          for (auto ind : Region3D(VectorI(0), VectorI(24))) {
            auto ab = an[VectorI(ind) / VectorI(Block::size) - VectorI(1)];
            if (!ab) {
              u.node(ind) = 0;
              v.node(ind) = 0;
              w.node(ind) = 0;
              rho.node(ind) = 0;
              T.node(ind) = 0;
            } else {
              auto node = ab->node_local(VectorI(ind) % VectorI(Block::size));
              u.node(ind) = node[CH_VX];
              v.node(ind) = node[CH_VY];
              w.node(ind) = node[CH_VZ];
              rho.node(ind) = node[CH_RHO];
              T.node(ind) = node[CH_T];
            }
          }

          auto sample_velocity = [&](Vector3 pos) -> Vector3 {
            return Vector3(u.sample(pos), v.sample(pos), w.sample(pos));
          };
          auto backtrace = [&](Vector3 pos) {
            // RK2
            return pos -
                   dt * sample_velocity(pos -
                                        (dt * 0.5_f) * sample_velocity(pos));
          };
          auto offset = Vector3i(Block::size);
          for (auto ind : b.local_region()) {
            auto &node = b.node_local(ind);
            node[CH_VX] = u.sample(backtrace(u.node_pos(ind + offset)));
            node[CH_VY] = v.sample(backtrace(v.node_pos(ind + offset)));
            node[CH_VZ] = w.sample(backtrace(w.node_pos(ind + offset)));
            node[CH_RHO] = rho.sample(backtrace(rho.node_pos(ind + offset)));
            node[CH_T] = T.sample(backtrace(T.node_pos(ind + offset)));
          }

          if (b.meta.get_has_effective_cell()) {
            Vector particle_range[] = {
                b.base_coord.template cast<real>() * dx,
                (b.base_coord + VectorI(Block::size)).template cast<real>() *
                    dx};
            // Gather and move particles
            for (auto ab : an.data) {
              if (ab) {
                for (std::size_t i = 0; i < ab->particle_count; i++) {
                  // Copy
                  auto p = ab->particles[i];
                  if (particle_range[0] < p.pos && p.pos < particle_range[1]) {
                    p.pos += sample_velocity(p.pos) * dt;
                    b.add_particle(p);
                  }
                }
              }
            }
          }
        },
        false);
  }

  void compute_b(bool debug = false) {
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          Grid::GridScratchPad scratch(an);
          for (auto &ind : b.local_region()) {
            auto center = VectorI(ind);
            auto div = 0.0_f;
            for (int i = 0; i < dim; i++) {
              div += scratch.node(center)[CH_VX + i] -
                     scratch.node(center + VectorI::axis(i))[CH_VX + i];
            }
            b.node_local(ind)[CH_B] = div;
            if (debug && std::abs(div) > 1e-3_f) {
              TC_P(b.base_coord + ind);
              TC_P(div);
            }
          }
        },
        false, true);
  }

  void project() {
    // Compute divergence
    compute_b();
    real before_projection = norm(CH_B);
    TC_P(before_projection);
    // Solve Poisson
    poisson_solve();
    // Apply pressure
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          Grid::GridScratchPad scratch(an);
          for (auto &ind : b.local_region()) {
            auto center = VectorI(ind);
            for (int k = 0; k < dim; k++) {
              b.node_local(ind)[CH_VX + k] -=
                  scratch.node(center)[CH_X] -
                  scratch.node(center - VectorI::axis(k))[CH_X];
            }
          }

        },
        false, true);
    if (false) {
      compute_b(true);
      real after_projection = norm(CH_B);
      if (after_projection > 1e-4_f) {
        TC_WARN("After projection: {}", after_projection);
      }
    }
  }

  void enforce_boundary_condition() {
    grids[0]->map([&](Block &b) {
      if (current_t == 0) {
        for (auto &ind : b.local_region()) {
          b.node_local(ind)[CH_VX] = 0;
          b.node_local(ind)[CH_VY] = 0;
          b.node_local(ind)[CH_VZ] = 0;
          b.node_local(ind)[CH_RHO] = 0;
          b.node_local(ind)[CH_T] = 0;
        }
      }
      if (b.base_coord == VectorI(0, -n * 2 + 8, 0)) {
        // Sample some particles
        for (int i = 0; i < 3e5 * dt; i++) {
          auto r = Vector::rand();
          if (length(r - Vector(0.5_f)) > 0.5) {
            continue;
          }
          Vector pos = (b.base_coord.template cast<real>() +
                        r * VectorI(Block::size).template cast<real>()) *
                       dx;
          pos[3] = current_t;
          b.add_particle(Particle{pos});
        }
        // if (current_t == 0) {
        for (auto ind : b.local_region()) {
          b.node_local(ind)[CH_VX] = std::sin(current_t * 50);
          b.node_local(ind)[CH_VY] = 0;
          b.node_local(ind)[CH_VZ] = 0;
          b.node_local(ind)[CH_RHO] = 1;
          b.node_local(ind)[CH_T] = (std::cos(current_t * 30)) * 0.2_f + 1_f;
        }
      }
      real scale = std::exp(-temperature_decay * dt);
      for (auto ind : b.local_region()) {
        b.node_local(ind)[CH_VY] += (b.node_local(ind)[CH_T] * buoyancy -
                                     b.node_local(ind)[CH_RHO] * gravity) *
                                    dt;
        b.node_local(ind)[CH_T] *= scale;
      }
    });
  }

  void render(Canvas &canvas) {
    auto res = canvas.img.get_res();
    Array2D<Vector3> image(Vector2i(res),
                           Vector3(0.7, 0.7, 0.7) - Vector3(0.7_f));
    std::vector<RenderParticle> particles;
    for (auto &p : grids[0]->gather_particles()) {
      auto t = p.pos[3];
      auto color = hsv2rgb(Vector(fract(t) * 2, 0.7_f, 0.9_f));
      particles.push_back(
          RenderParticle(p.pos * Vector(0.16_f), Vector4(color, 1.0_f)));
    }
    renderer->render(image, particles);
    for (auto &ind : image.get_region()) {
      canvas.img[ind] = Vector4(image[ind]);
    }
  }

  void step() {
    TC_PROFILE("BC", enforce_boundary_condition());
    TC_PROFILE("Advection", advect());
    TC_PROFILE("Project", project());
    current_t += dt;
  }

  void test_renderer() {
    int res = 800;
    GUI gui("Rendering Test", res, res);
    Array2D<Vector3> image(Vector2i(res), Vector3(0.5));
    std::vector<RenderParticle> particles;
    for (int i = 0; i < 1000000; i++) {
      Vector3 pos = Vector::rand() - Vector3(0.5_f);
      particles.push_back(RenderParticle(pos * Vector(10_f),
                                         Vector4(1.0_f, 1.0_f, 0.0_f, 0.5_f)));
    }
    renderer->render(image, particles);
    auto &canvas = gui.get_canvas().img;
    for (auto &ind : image.get_region()) {
      canvas[ind] = Vector4(image[ind]);
    }
    while (1) {
      gui.update();
    }
  }

  Array2D<Vector3> render_density_field() {
    Array2D<Vector3> img;
    img.initialize(Vector2i(n * 2, n * 4));
    grids[0]->for_each_block([&](Block &b) {
      if (b.base_coord.z != 0 || !b.meta.get_has_effective_cell()) {
        return;
      }
      for (int i = 0; i < b.size[0]; i++) {
        for (int j = 0; j < b.size[1]; j++) {
          auto node = b.node_local(Vector3i(i, j, 0));
          auto vel = Vector3(node[CH_RHO]);
          img[Vector2i(b.base_coord.x, b.base_coord.y) +
              Vector2i(n + i, n * 2 + j)] = vel * Vector3(1);
        }
      }
    });
    return img;
  }

  Array2D<Vector3> render_velocity_field() {
    Array2D<Vector3> img;
    img.initialize(Vector2i(n * 2, n * 4));
    grids[0]->for_each_block([&](Block &b) {
      if (b.base_coord.z != 0 || !b.meta.get_has_effective_cell()) {
        return;
      }
      for (int i = 0; i < b.size[0]; i++) {
        for (int j = 0; j < b.size[1]; j++) {
          auto node = b.node_local(Vector3i(i, j, 0));
          auto vel = Vector3(node[CH_VX], node[CH_VY], node[CH_VZ]);
          img[Vector2i(b.base_coord.x, b.base_coord.y) +
              Vector2i(n + i, n * 2 + j)] = vel * Vector3(1) + Vector3(0.5_f);
        }
      }
    });
    return img;
  }

  Array2D<Vector3> render_pressure_field() {
    Array2D<Vector3> img;
    img.initialize(Vector2i(n * 2, n * 4));
    grids[0]->for_each_block([&](Block &b) {
      if (b.base_coord.z != 0 || !b.meta.get_has_effective_cell()) {
        return;
      }
      for (int i = 0; i < b.size[0]; i++) {
        for (int j = 0; j < b.size[1]; j++) {
          auto node = b.node_local(Vector3i(i, j, 0));
          auto vel = Vector3(node[CH_X]);
          img[Vector2i(b.base_coord.x, b.base_coord.y) +
              Vector2i(n + i, n * 2 + j)] = vel * Vector3(1) + Vector3(0.5_f);
        }
      }
    });
    return img;
  }
};

auto mgpcg = [](const std::vector<std::string> &params) {
  ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MGPCGSmoke> mgpcg;
  mgpcg = std::make_unique<MGPCGSmoke>();
  while (true) {
    TC_TIME(mgpcg->test_pcg());
    print_profile_info();
  }
  GUI gui2("Pressure", 256, 512);
  auto img = mgpcg->render_pressure_field();
  for (auto ind : gui2.get_canvas().img.get_region()) {
    gui2.get_canvas().img[ind] = Vector3(img[Vector2i(ind) / Vector2i(4)]);
  }
  while (1)
    gui2.update();
};

TC_REGISTER_TASK(mgpcg);

auto smoke = [](const std::vector<std::string> &params) {
  TC_PROFILER("smoke");
  // ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MGPCGSmoke> smoke;
  smoke = std::make_unique<MGPCGSmoke>();
  GUI gui("MGPCG Smoke", 800, 800);
  // GUI gui2("Velocity", 256, 512);
  // GUI gui3("Density", 256, 512);

  for (int frame = 0;; frame++) {
    TC_PROFILE("step", smoke->step());
    gui.get_canvas().img.write_as_image(fmt::format("tmp/{:05d}.png", frame));
    TC_PROFILE("render", smoke->render(gui.get_canvas()));
    gui.update();
    {
      /*
      TC_PROFILER("debug");
      auto img = smoke->render_velocity_field();
      for (auto ind : gui2.get_canvas().img.get_region()) {
        gui2.get_canvas().img[ind] = Vector3(img[Vector2i(ind) / Vector2i(4)]);
      }
      gui2.update();
      img = smoke->render_density_field();
      for (auto ind : gui3.get_canvas().img.get_region()) {
        gui3.get_canvas().img[ind] = Vector3(img[Vector2i(ind) / Vector2i(4)]);
      }
      gui3.update();
      */
    }
    print_profile_info();
  }
};

TC_REGISTER_TASK(smoke);

auto test_volume_rendering = [](const std::vector<std::string> &params) {
  std::unique_ptr<MGPCGSmoke> smoke;
  smoke = std::make_unique<MGPCGSmoke>();
  smoke->test_renderer();
};

TC_REGISTER_TASK(test_volume_rendering);

TC_NAMESPACE_END
