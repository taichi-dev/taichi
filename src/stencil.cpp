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
  auto left = Input<0, Offset<0, 0, -1>>();
  auto right = Input<1, Offset<0, 0, 1>>();
  auto top = Input<1, Offset<0, 1, 0>>();
  auto bottom = Input<0, Offset<0, -1, 0>>();

}

TC_NAMESPACE_END
