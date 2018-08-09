#include "taichi_grid.h"
#include <taichi/common/meta.h>
#include <tuple>

TC_NAMESPACE_BEGIN

namespace stencilang {

template <int _i, int _j, int _k>
struct Offset {
  static constexpr int i = _i;
  static constexpr int j = _j;
  static constexpr int k = _k;
};

template <typename _a, typename _b>
struct Add {
  using a = _a;
  using b = _b;

  static std::string serialize() {
    return a::serialize() + " + " + b::serialize();
  }

  // args:
  // 0: linearized based (output) address
  // 1, ...: channels

  template <typename... Args>
  TC_FORCE_INLINE static auto evaluate(Args const &... args) {
    auto ret_a = a::evaluate(args...);
    auto ret_b = b::evaluate(args...);
    return _mm256_add_ps(ret_a, ret_b);
  }
};

template <int _channel, typename _offset>
struct Input {
  static constexpr int channel = _channel;
  using offset = _offset;

  template <typename O>
  auto operator+(const O &o) {
    return Add<Input, O>();
  }

  static std::string serialize() {
    return fmt::format("D{}({:+},{:+},{:+})", channel, offset::i, offset::j,
                       offset::k);
  };

  template <typename... Args>
  TC_FORCE_INLINE static __m256 evaluate(Args const &... args) {
    int base = std::get<0>(std::tuple<Args const &...>(args...));
    using pad_type = typename std::decay_t<decltype(
        std::get<channel + 1>(std::tuple<Args const &...>(args...)))>;
    auto const &pad =
        std::get<channel + 1>(std::tuple<Args const &...>(args...));
    constexpr int offset =
        pad_type::template relative_offset<offset::i, offset::j, offset::k>();
    return _mm256_loadu_ps(&pad.linearized_data[base + offset]);
  }
};
}

struct Node {
  constexpr static int num_channels = 16;
  using element_type = real;
  /*
  NodeFlags &flags() {
    return bit::reinterpret_bits<NodeFlags>(channels[15]);
  }
  */
};

template <>
constexpr bool is_SOA<Node>() {
  return true;
}
using Block = TBlock<Node, char, TSize3D<8>, 0, 1>;

auto stencil = [](const std::vector<std::string> &params) {
  using namespace stencilang;
  auto left = Input<0, Offset<0, 0, -1>>();
  auto right = Input<0, Offset<0, 0, 1>>();
  auto sum = left + right;
  TC_P(sum.serialize());

  using GridScratchPadCh = TGridScratchPad<Block, real>;
  using GridScratchPadCh2 = TGridScratchPad<Block, real, 2>;

  GridScratchPadCh2 pad;
  for (auto ind : pad.region()) {
    pad.node(ind) = ind.k;
  }

  float32 _ret[8];
  auto ret = (__m256 *)&_ret[0];
  *ret = sum.evaluate(1, pad);
  TC_P(_ret);
};

TC_REGISTER_TASK(stencil);

TC_NAMESPACE_END
