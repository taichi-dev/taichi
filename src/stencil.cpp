#include "stencil.h"

TC_NAMESPACE_BEGIN

struct Node {
  constexpr static int num_channels = 16;
  using element_type = real;
};

template <>
constexpr bool is_SOA<Node>() {
  return true;
}
using Block = TBlock<Node, char, TSize3D<8>, 0, 1>;

auto stencil = [](const std::vector<std::string> &params) {
  using namespace stencilang;
  auto left = Input<0, Offset<0, 0, -1>>();
  auto right = Input<1, Offset<0, 0, 1>>();
  auto top = Input<1, Offset<0, 1, 0>>();
  auto bottom = Input<0, Offset<0, -1, 0>>();
  auto sum = (left + right) * top + Ratio<1, 3>();
  TC_P(sum.serialize());

  using GridScratchPadCh = TGridScratchPad<Block, real>;
  using GridScratchPadCh2 = TGridScratchPad<Block, real, 2>;

  GridScratchPadCh2 pad1;
  for (auto ind : pad1.region()) {
    pad1.node(ind) = ind.k;
  }

  GridScratchPadCh2 pad2;
  for (auto ind : pad1.region()) {
    pad2.node(ind) = -ind.k;
  }

  float32 _ret[8];
  auto ret = (__m256 *)&_ret[0];
  int base = GridScratchPadCh2::linear_offset<1, 1, 1>();
  *ret = sum.evaluate(base, pad1, pad2);
  TC_P(_ret);
};

TC_REGISTER_TASK(stencil);

TC_TEST("stencil") {
  using namespace stencilang;
  constexpr int ChU = 0;

  using Scratch = TGridScratchPad<Block, real, 2>;

  Scratch scratchU;
  Scratch scratchV;  // For iteration

  auto c = sizeof(Scratch::linearized_data) / sizeof(real);
  for (int i = 0; i < c; i++) {
    auto d = rand();
    scratchU.linearized_data[i] = d;
    TC_CHECK(d == d);
  }

  Scratch _scratchU;
  std::memcpy(_scratchU.linearized_data, scratchU.linearized_data,
              sizeof(scratchU.linearized_data));
  Scratch _scratchV;  // For iteration

  // clang-format off
  auto sum =
      (input<ChU, Offset<0, 0, 1>> + input<ChU, Offset<0, 0, -1>>) +
      (input<ChU, Offset<0, 1, 0>> + input<ChU, Offset<0, -1, 0>>) +
      (input<ChU, Offset<1, 0, 0>> + input<ChU, Offset<-1, 0, 0>>);
  auto jacobi = sum * ratio<1, 6>;

  map(_scratchV, jacobi,
      Region3D(Vector3i(-1), Vector3i(Block::size) + Vector3i(1)),
      _scratchU);
  map(_scratchU, jacobi,
      Region3D(Vector3i(0), Vector3i(Block::size) + Vector3i(0)),
      _scratchV);

  for (int i = -1; i < Scratch::block_size[0] + 1; i++) {
    for (int j = -1; j < Scratch::block_size[1] + 1; j++) {
      for (int k = -1; k < Scratch::block_size[2] + 1; k++) {
        scratchV.data[i][j][k] = 1.0_f / 6 * (
                                                 scratchU.data[i][j][k - 1] +
                                                     scratchU.data[i][j][k + 1] +
                                                     scratchU.data[i][j - 1][k] +
                                                     scratchU.data[i][j + 1][k] +
                                                     scratchU.data[i - 1][j][k] +
                                                     scratchU.data[i + 1][j][k]
                                             );
      }
    }
  }
  for (int i = 0; i < Scratch::block_size[0]; i++) {
    for (int j = 0; j < Scratch::block_size[1]; j++) {
      for (int k = 0; k < Scratch::block_size[2]; k++) {
        auto ret = 1.0_f / 6 * (
            scratchV.data[i][j][k - 1] +
            scratchV.data[i][j][k + 1] +
            scratchV.data[i][j - 1][k] +
            scratchV.data[i][j + 1][k] +
            scratchV.data[i - 1][j][k] +
            scratchV.data[i + 1][j][k]
        );
        TC_CHECK_EQUAL(ret, _scratchU.data[i][j][k], 1e-4_f);
        //TC_CHECK(ret == _scratchU.data[i][j][k]);
      }
    }
  }
}

TC_NAMESPACE_END
