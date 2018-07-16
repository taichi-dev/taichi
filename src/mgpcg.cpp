#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>
#include <taichi/util.h>
#include <taichi/math/svd.h>

TC_NAMESPACE_BEGIN

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

using Block = TBlock<Node, char, TSize3D<8>, 0>;

class MGPCGTest {
 public:
  static constexpr auto dim = Block::dim;
  using Vector = TVector<real, dim>;
  using VectorP = TVector<real, dim + 1>;
  using Matrix = TMatrix<real, dim>;
  using Grid = TaichiGrid<Block>;

  std::vector<std::unique_ptr<Grid>> grids;
  int mg_lv = 2;

  using VectorI = Vector3i;
  using Vectori = VectorI;
  using GridScratchPad = TGridScratchPad<Block>;

  enum { CH_R, CH_Z, CH_X, CH_B, CH_TMP, CH_P, CH_MG_U, CH_MG_B, CH_MG_R };

  MGPCGTest() {
    // Span a region in
    grids.resize(mg_lv);
    for (int i = 0; i < mg_lv; i++) {
      grids[i] = std::make_unique<Grid>();
    }
    TC_ASSERT(mg_lv >= 1);
    constexpr int n = 32;
    TC_ASSERT_INFO(bit::is_power_of_two(n), "Only POT grid sizes supported");
    Region3D active_region(VectorI(-n, -n, -n * 2), VectorI(n, n, n * 2));
    for (auto &ind : active_region) {
      grids[0]->touch(ind.get_ipos());
      grids[0]->node(ind.get_ipos()).flags().set_effective(true);
      if (ind.get_ipos() == VectorI(0)) {
        grids[0]->node(ind.get_ipos())[CH_B] = 1;
      }
    }
    set_up_hierechy();
  }

  void set_up_hierechy() {
    int total_blocks = grids[0]->num_active_blocks();
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
          std::memcpy(&b.nodes[0], &an[VectorI(0)]->nodes[0], sizeof(b.nodes));
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
        false);
  }

  void multiply(int channel_out, int channel_in) {
    // TODO: this supports zero-Dirichlet BC only!
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          GridScratchPad scratch(an);
          std::memcpy(&b.nodes[0], &an[VectorI(0)]->nodes[0], sizeof(b.nodes));
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
        false);
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
          std::memcpy(&b.nodes[0], &an[VectorI(0)]->nodes[0], sizeof(b.nodes));
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
                o = original + (tmp / count - original) * 1;  // 0.666667_f;
                /*
                if (tmp != 0) {
                  TC_P(b.base_coord + Vector3i(i, j, k));
                  TC_P(o);
                  TC_P(tmp);
                  TC_P(count);
                }
                */
              }
            }
          }
        },
        false);
  }

  void clear(int level, int channel) {
    grids[level]->for_each_node([&](Block::Node &n) { n[channel] = 0; });
  }

  // B[level + 1] = coarsened(R[level])
  void restrict(int level, int R_in, int B_out) {
    // NOTE: supports POT grids only
    // average residual
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
            /*
            if (correction != 0) {
              TC_P(correction);
            }
            */
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
    constexpr int smoothing_iters = 1, bottom_smoothing_iter = 10;
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
    for (int i = 0; i < 100; i++) {
      V_cycle(CH_B, CH_X, false);
      residual(0, CH_X, CH_B, CH_R);
      /*
      grids[0]->for_each_block([&](Block &b) {
        for (auto ind : b.get_global_region())  {
          auto val = b.node_global(ind.get_ipos())[CH_B];
          if ()
        }
      });
      */
      TC_P(norm(CH_R));
    }
  }

  // https://en.wikipedia.org/wiki/Conjugate_gradient_method
  void run_pcg() {
    TC_P(norm(CH_B));
    // r = b - Ax
    multiply(CH_TMP, CH_X);
    TC_P(norm(CH_TMP));
    saxpy(CH_R, CH_B, CH_TMP, -1);
    // z = M^-1 r
    saxpy(CH_Z, CH_R, CH_R, 0);
    // p = z
    copy(CH_P, CH_Z);
    while (1) {
      multiply(CH_TMP, CH_P);
      real alpha = dot_product(CH_R, CH_Z) / dot_product(CH_P, CH_TMP);

      auto old_zr = dot_product(CH_Z, CH_R);

      saxpy(CH_X, CH_X, CH_P, alpha);

      saxpy(CH_R, CH_R, CH_TMP, -alpha);

      auto l2 = norm(CH_R);
      TC_P(l2);
      if (l2 < 1e-7) {
        break;
      }

      copy(CH_Z, CH_R);

      auto beta = dot_product(CH_Z, CH_R) / old_zr;

      saxpy(CH_P, CH_Z, CH_P, beta);
    }
  }
};

auto mgpcg = [](const std::vector<std::string> &params) {
  ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MGPCGTest> mgpcg;
  mgpcg = std::make_unique<MGPCGTest>();
  mgpcg->run();
};

TC_REGISTER_TASK(mgpcg);

TC_NAMESPACE_END
