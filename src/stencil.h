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
    return "(" + a::serialize() + " + " + b::serialize() + ")";
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
    return "(" + a::serialize() + " * " + b::serialize() + ")";
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
    return "(" + a::serialize() + " - " + b::serialize() + ")";
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

template <int a, int b>
struct Ratio : public OpBase<Ratio<a, b>> {
  static std::string serialize() {
    return fmt::format("({}/{})", a, b);
  };

  template <typename... Args>
  TC_FORCE_INLINE static __m256 evaluate(Args const &... args) {
    auto constexpr val = (float32)a / b;
    return _mm256_set1_ps(val);
  }
};
}

template <typename Op, typename Output, typename... Args>
void map(Output &output, TRegion<3> region, Args const &... args) {
  int start = Output::linear_offset(region.begin());
  int end = Output::linear_offset(region.end());
  for (int i = start; i < end; i += 8) {
    auto ret = Op::evaluate(i, args...);
    _mm256_storeu_ps(&output.linearized_data[i], ret);
  }
}

TC_NAMESPACE_END
