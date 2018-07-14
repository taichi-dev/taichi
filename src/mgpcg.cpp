#include "taichi_grid.h"
#include <taichi/visual/texture.h>
#include <taichi/system/threading.h>
#include <taichi/util.h>
#include <taichi/math/svd.h>

TC_NAMESPACE_BEGIN

struct Node {
  real channels[16];
  real &operator[](int i) {
    return channels[i];
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
  int mg_lv = 4;

  using VectorI = Vector3i;
  using Vectori = VectorI;
  using GridScratchPad = TGridScratchPad<Block>;

  enum { CH_R, CH_Z, CH_X, CH_B, CH_TMP, CH_P };

  MGPCGTest() {
    // Span a region in
    grids.resize(mg_lv);
    for (int i = 0; i < mg_lv; i++) {
      grids[i] = std::make_unique<Grid>();
    }
    TC_ASSERT(mg_lv >= 1);
    constexpr int n = 32;
    Region3D active_region(VectorI(-n, -n, -n * 2), VectorI(n, n, n * 2));
    for (auto &ind : active_region) {
      grids[0]->touch(ind.get_ipos());
      if (ind.get_ipos() == VectorI(0)) {
        TC_TAG;
        grids[0]->node(ind.get_ipos())[CH_B] = 1;
      }
    }
  }

  void multiply(int channel_out, int channel_in) {
    grids[0]->advance(
        [&](Block &b, Grid::Ancestors &an) {
          GridScratchPad scratch(an);
          std::memcpy(&b.nodes[0], &an[VectorI(0)]->nodes[0], sizeof(b.nodes));
          // 6 neighbours
          for (int i = 0; i < Block::size[0]; i++) {
            for (int j = 0; j < Block::size[1]; j++) {
              for (int k = 0; k < Block::size[2]; k++) {
#define V(ii, jj, kk) scratch.data[i + (ii)][j + (jj)][k + (kk)][channel_in]
                auto &o = b.get_node_volume()[i][j][k][channel_out];
                o = 6 * V(0, 0, 0) - V(0, 0, 1) - V(0, 0, -1) - V(0, 1, 0) -
                    V(0, -1, 0) - V(1, 0, 0) - V(-1, 0, 0);
                if (o != o) {
                  TC_P(b.base_coord);
                  TC_P(V(0, 0, 0));
                  TC_P(V(0, 0, 1));
                  TC_P(V(0, 1, 0));
                  TC_P(V(1, 0, 0));
                  TC_P(V(0, 0, -1));
                  TC_P(V(0, -1, 0));
                  TC_P(V(-1, 0, 0));
                  TC_P(o);
                  TC_P(i);
                  TC_P(j);
                  TC_P(k);
                  Time::sleep(0.01);
                }
#undef V
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

  void smoothing() {
    grids[0]->advance(
        [&](Grid::Block &b, Grid::Ancestors &an) {

        },
        true);
  }

  real norm(int channel) {
    return std::sqrt(dot_product(channel, channel));
  }

  // https://en.wikipedia.org/wiki/Conjugate_gradient_method
  void run() {
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
  // ThreadedTaskManager::TbbParallelismControl _(1);
  std::unique_ptr<MGPCGTest> mgpcg;
  mgpcg = std::make_unique<MGPCGTest>();
  mgpcg->run();
};

TC_REGISTER_TASK(mgpcg);

TC_NAMESPACE_END
