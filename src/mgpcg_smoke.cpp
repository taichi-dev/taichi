#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>
#include <taichi/util.h>
#include <taichi/math/svd.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/gui.h>

TC_NAMESPACE_BEGIN

// TODO: u, v, w has different sizes

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

using Block = TBlock<Node, Particle, TSize3D<8>, 0>;

class MGPCGSmoke {
 public:
  static constexpr auto dim = Block::dim;
  using Vector = TVector<real, dim>;
  using VectorP = TVector<real, dim + 1>;
  using Matrix = TMatrix<real, dim>;
  using Grid = TaichiGrid<Block>;

  std::vector<std::unique_ptr<Grid>> grids;
  static constexpr int mg_lv = 3;

  using VectorI = Vector3i;
  using Vectori = VectorI;
  using GridScratchPad = TGridScratchPad<Block>;

  const int n = 32;

  std::shared_ptr<Camera> cam;
  real current_t;
  real dt = 1e-2_f, dx = 1.0_f / n, inv_dx = 1.0_f / dx;

  std::unique_ptr<ParticleRenderer> renderer;

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
    CH_DENSITY
  };

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
    dict.set("shadow_map_resolution", 0.5_f)
        .set("alpha", 0.6_f)
        .set("shadowing", 0.07_f)
        .set("ambient_light", 0.3_f)
        .set("light_direction", Vector(1, 3, 1));
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
    Region3D active_region(VectorI(-n, -n, -n * 2), VectorI(n, n, n * 2));
    for (auto ind : active_region) {
      grids[0]->touch(ind);
      grids[0]->node(ind).flags().set_effective(true);
    }
    set_up_hierechy();
  }

  void test_pcg() {
    Region3D active_region(VectorI(-n, -n, -n * 2), VectorI(n, n, n * 2));
    for (auto &ind : active_region) {
      if (ind.get_ipos() == VectorI(0)) {
        grids[0]->node(ind.get_ipos())[CH_B] = 1;
      }
    }
    poisson_solve();
  }

  void set_up_hierechy() {
    std::size_t total_blocks = grids[0]->num_active_blocks();
    for (int i = 0; i < mg_lv - 1; i++) {
      grids[i]->coarsen_to(
          *grids[i + 1], [&](Block &b, Grid::PyramidAncestors &an) {
            for (auto ind : b.get_local_region()) {
              b.node_local(ind.get_ipos()).flags().set_effective(true);
            }
          });
      total_blocks /= 8;
      TC_ASSERT(grids[i + 1]->num_active_blocks() == total_blocks);
    }
  }

  void residual(int level, int U, int B, int R) {
    grids[level]->advance(
        [&](Block &b, Grid::Ancestors &an) {
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
    // TODO: this supports zero-Dirichlet BC only!
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          GridScratchPad scratch(an);
          // 6 neighbours
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              for (int k = 0; k < Block::size[2]; k++) {
                int count = 0;
                real tmp = 0;
                auto &o = b.get_node_volume()[i][j][k][channel_out];
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
                o = count * scratch.data[i][j][k][channel_in] - tmp;
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
    // TODO: this supports zero-Dirichlet BC only!
    grids[level]->advance(
        [&](Grid::Block &b, Grid::Ancestors &an) {
          GridScratchPad scratch(an);
          // 6 neighbours
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              for (int k = 0; k < Block::size[2]; k++) {
                TC_ASSERT(scratch.data[i][j][k].flags().get_effective());
                int count = 0;
                // (B - Lu) / Diag
                real tmp = scratch.data[i][j][k][B];
                auto fetch = [&](int ii, int jj, int kk) {
                  count += 1;
                  tmp += scratch.data[i + ii][j + jj][k + kk][U];
                };
                fetch(0, 0, 1);
                fetch(0, 0, -1);
                fetch(0, 1, 0);
                fetch(0, -1, 0);
                fetch(1, 0, 0);
                fetch(-1, 0, 0);
                auto original = scratch.data[i][j][k][U];
                TC_ASSERT(count != 0);
                auto &o = b.get_node_volume()[i][j][k][U];
                // Damping is important. It brings down #iterations to 1e-7 from
                // 91 to 10...
                o = original + (tmp / count - original) * (2.0_f / 3_f);
              }
            }
          }
        },
        false, true);
  }

  void clear(int level, int channel) {
    grids[level]->for_each_node([&](Block::Node &n) { n[channel] = 0; });
  }

  // B[level + 1] = coarsened(R[level])
  void restrict(int level, int R_in, int B_out) {
    // NOTE: supports POT grids only
    // sum residual
    clear(level + 1, B_out);
    grids[level]->coarsen_to(
        *grids[level + 1], [&](Block &block, Grid::PyramidAncestors &an) {
          for (auto ind : Region3D(Vector3i(0), Vector3i(2))) {
            if (!an[ind.get_ipos()]) {
              TC_NOT_IMPLEMENTED
              continue;
            }
            Block &ab = *an[ind.get_ipos()];
            for (auto j : ab.get_local_region()) {
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
    real scale = 0.5_f;
    // upsample and apply correction
    grids[level]->refine_from(
        *grids[level + 1], [&](Block &block, Block &ancestor) {
          for (auto ind : block.get_global_region()) {
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
    constexpr int smoothing_iters = 3, bottom_smoothing_iter = 50;
    if (use_as_preconditioner) {
      clear(0, U);
    }
    for (int i = 0; i < mg_lv - 1; i++) {
      // pre-smoothing
      for (int j = 0; j < smoothing_iters; j++) {
        smooth(i, U, B);
      }
      residual(i, U, B, R);
      restrict(i, R, B);
      clear(i + 1, U);
    }

    // Bottom solve
    for (int j = 0; j < bottom_smoothing_iter; j++) {
      smooth(mg_lv - 1, U, B);
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
    constexpr real tolerance = 1e-4_f;
    bool use_preconditioner = true;
    real initial_residual_norm = norm(CH_B);
    TC_P(initial_residual_norm);
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
      multiply(CH_Z, CH_P);
      real alpha = old_zr / dot_product(CH_P, CH_Z);

      saxpy(CH_X, CH_X, CH_P, alpha);
      saxpy(CH_R, CH_R, CH_Z, -alpha);

      auto residual_l2 = norm(CH_R);
      TC_TRACE("iter {}, residual {}", i, residual_l2);
      if (residual_l2 < tolerance * initial_residual_norm) {
        break;
      }

      if (use_preconditioner) {
        V_cycle(CH_R, CH_Z);
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
    return axis < 3 ? Vector3::axis(axis) * 0.5_f : Vector3(0.5_f);
  }

  void advect() {
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          TC_STATIC_ASSERT(Block::size[0] == 8);
          auto scale = Vector(inv_dx);
          auto corner = b.base_coord.template cast<real>() -
                        VectorI(Block::size).template cast<real>();

          // Unfortunately, ux, uy, uz and density are not collocated...
          LerpField<real, TSize3D<24>> u(scale,
                                         corner + Vector(0.5_f, 0.0_f, 0.0_f));
          LerpField<real, TSize3D<24>> v(scale,
                                         corner + Vector(0.0_f, 0.5_f, 0.0_f));
          LerpField<real, TSize3D<24>> w(scale,
                                         corner + Vector(0.0_f, 0.0_f, 0.5_f));
          LerpField<real, TSize3D<24>> rho(
              scale, corner + Vector(0.5_f, 0.5_f, 0.5_f));

          for (auto ind : Region3D(VectorI(0), VectorI(24))) {
            auto ab = an[VectorI(ind) / VectorI(Block::size) - VectorI(1)];
            if (!ab) {
              u.node(ind) = 0;
              v.node(ind) = 0;
              w.node(ind) = 0;
              rho.node(ind) = 0;
            } else {
              auto node = ab->node_local(VectorI(ind) % VectorI(Block::size));
              u.node(ind) = node[CH_VX];
              v.node(ind) = node[CH_VY];
              w.node(ind) = node[CH_VZ];
              rho.node(ind) = node[CH_DENSITY];
            }
          }

          // TODO: fix local/global

          auto sample_velocity = [&](Vector3 pos) -> Vector3 {
            return Vector3(u.sample(pos), v.sample(pos), w.sample(pos));
          };
          auto backtrace = [&](Vector3 pos) {
            // RK2
            return pos -
                   dt * sample_velocity(pos -
                                        (dt * 0.5_f) * sample_velocity(pos));
          };
          for (auto ind : b.get_local_region()) {
            auto node = b.node_local(ind.get_ipos());
            node[CH_VX + 0] = u.sample(backtrace(u.node_pos(ind)));
            node[CH_VX + 1] = v.sample(backtrace(v.node_pos(ind)));
            node[CH_VX + 2] = w.sample(backtrace(w.node_pos(ind)));
            node[CH_VX + 3] = rho.sample_global(backtrace(rho.node_pos(ind)));
          }

          Vector particle_range[] = {
              b.base_coord.template cast<real>() * dx,
              (b.base_coord + VectorI(Block::size)).template cast<real>() * dx};
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
        },
        false);
  }

  void compute_b() {
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          Grid::GridScratchPad scratch(an);
          for (auto &ind : b.get_local_region()) {
            auto center = VectorI(ind);
            auto div = 0;
            for (int i = 0; i < 3; i++) {
              div += scratch.node(center)[CH_VX + i] -
                     scratch.node(center + VectorI::axis(i))[CH_VX + i];
            }
            b.node_local(ind)[CH_B] = div;
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
          for (auto &ind : b.get_local_region()) {
            auto center = VectorI(ind);
            for (int k = 0; k < dim; k++) {
              b.node_local(ind)[CH_VX + k] -=
                  scratch.node(center)[CH_X] -
                  scratch.node(center - VectorI::axis(k))[CH_X];
            }
          }

        },
        false, true);
    compute_b();
    real after_projection = norm(CH_B);
    TC_WARN("After projection: {}", after_projection);
  }

  void enforce_boundary_condition() {
    grids[0]->map([&](Block &b) {
      if (current_t == 0) {
        for (auto &ind : b.get_local_region()) {
          b.node_local(ind)[CH_VX] = 0;
          b.node_local(ind)[CH_VY] = 1;
          b.node_local(ind)[CH_VZ] = 0;
        }
      }
      if (b.base_coord == VectorI(0)) {
        // Sample some particles
        for (int i = 0; i < 100; i++) {
          Vector pos =
              (b.base_coord.template cast<real>() +
               Vector::rand() * VectorI(Block::size).template cast<real>()) *
              dx;
          b.add_particle(Particle{pos});
        }
      }
    });
  }

  void render(Canvas &canvas) {
    auto res = canvas.img.get_res();
    Array2D<Vector3> image(Vector2i(res), Vector3(0.7));
    std::vector<RenderParticle> particles;
    for (auto &p : grids[0]->gather_particles()) {
      particles.push_back(RenderParticle(p.pos * Vector(0.1_f),
                                         Vector4(1.0_f, 1.0_f, 0.0_f, 0.5_f)));
    }
    renderer->render(image, particles);
    for (auto &ind : image.get_region()) {
      canvas.img[ind] = Vector4(image[ind]);
    }
  }

  void step() {
    enforce_boundary_condition();
    advect();
    project();
    current_t += dt;
  }

  void test_renderer() {
    int res = 800;
    GUI gui("Rendering Test", res, res);
    Array2D<Vector3> image(Vector2i(res), Vector3(0.7));
    std::vector<RenderParticle> particles;
    for (int i = 0; i < 1000000; i++) {
      Vector3 pos = Vector::rand() - Vector3(0.5_f);
      particles.push_back(RenderParticle(pos * Vector(0.1_f),
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
};

auto mgpcg = [](const std::vector<std::string> &params) {
  // ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MGPCGSmoke> mgpcg;
  mgpcg = std::make_unique<MGPCGSmoke>();
  TC_TIME(mgpcg->test_pcg());
};

TC_REGISTER_TASK(mgpcg);

auto smoke = [](const std::vector<std::string> &params) {
  // ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MGPCGSmoke> smoke;
  smoke = std::make_unique<MGPCGSmoke>();
  GUI gui("MGPCG Smoke", 800, 800);
  while (1) {
    TC_TIME(smoke->step());
    smoke->render(gui.get_canvas());
    gui.update();
    return;
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
