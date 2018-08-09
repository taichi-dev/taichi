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
struct Add;

template <typename _a, typename _b>
struct Sub;

template <typename _a, typename _b>
struct Mul;

template <typename Op>
struct OpBase {
  template <typename O>
  auto operator+(const O &o) {
    return Add<Op, O>();
  }

  template <typename O>
  auto operator-(const O &o) {
    return Sub<Op, O>();
  }

  template <typename O>
  auto operator*(const O &o) {
    return Mul<Op, O>();
  }
};

template <typename _a, typename _b>
struct Add : public OpBase<Add<_a, _b>> {
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

template <typename _a, typename _b>
struct Mul : public OpBase<Mul<_a, _b>> {
  using a = _a;
  using b = _b;

  static std::string serialize() {
    return a::serialize() + " * " + b::serialize();
  }

  // args:
  // 0: linearized based (output) address
  // 1, ...: channels

  template <typename... Args>
  TC_FORCE_INLINE static auto evaluate(Args const &... args) {
    auto ret_a = a::evaluate(args...);
    auto ret_b = b::evaluate(args...);
    return _mm256_mul_ps(ret_a, ret_b);
  }
};

template <typename _a, typename _b>
struct Sub : public OpBase<Sub<_a, _b>> {
  using a = _a;
  using b = _b;

  static std::string serialize() {
    return a::serialize() + " - " + b::serialize();
  }

  // args:
  // 0: linearized based (output) address
  // 1, ...: channels

  template <typename... Args>
  TC_FORCE_INLINE static auto evaluate(Args const &... args) {
    auto ret_a = a::evaluate(args...);
    auto ret_b = b::evaluate(args...);
    return _mm256_sub_ps(ret_a, ret_b);
  }
};

template <int _channel, typename _offset>
struct Input : public OpBase<Input<_channel, _offset>> {
  static constexpr int channel = _channel;
  using offset = _offset;

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
  auto sum = (left + right) * top;
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

TC_NAMESPACE_END
