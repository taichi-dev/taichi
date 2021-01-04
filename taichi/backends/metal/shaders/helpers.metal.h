#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_HELPERS_DEF constexpr auto kMetalHelpersSourceCode =
#define METAL_END_HELPERS_DEF ;
#else
#define METAL_BEGIN_HELPERS_DEF
#define METAL_END_HELPERS_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

static_assert(false, "Do not include");

#define METAL_BEGIN_HELPERS_DEF
#define METAL_END_HELPERS_DEF

#include <type_traits>
using std::is_same;
using std::is_signed;

#endif  // TI_INSIDE_METAL_CODEGEN

// clang-format off
METAL_BEGIN_HELPERS_DEF
STR(
    // clang-format on
    template <typename T, typename G> T union_cast(G g) {
      // For some reason, if I emit taichi/common.h's union_cast(), Metal failed
      // to compile. More strangely, if I copy the generated code to XCode as a
      // Metal kernel, it compiled successfully...
      static_assert(sizeof(T) == sizeof(G), "Size mismatch");
      return *reinterpret_cast<thread const T *>(&g);
    }

    inline int ifloordiv(int lhs, int rhs) {
      const int intm = (lhs / rhs);
      return (((lhs < 0) != (rhs < 0) && lhs && (rhs * intm != lhs))
                  ? (intm - 1)
                  : intm);
    }

    int32_t pow_i32(int32_t x, int32_t n) {
      int32_t tmp = x;
      int32_t ans = 1;
      while (n) {
        if (n & 1) ans *= tmp;
        tmp *= tmp;
        n >>= 1;
      }
      return ans;
    }

    float fatomic_fetch_add(device float *dest, const float operand) {
      // A huge hack! Metal does not support atomic floating point numbers
      // natively.
      bool ok = false;
      float old_val = 0.0f;
      while (!ok) {
        old_val = *dest;
        float new_val = (old_val + operand);
        ok = atomic_compare_exchange_weak_explicit(
            (device atomic_int *)dest, (thread int *)(&old_val),
            *((thread int *)(&new_val)), metal::memory_order_relaxed,
            metal::memory_order_relaxed);
      }
      return old_val;
    }

    float fatomic_fetch_min(device float *dest, const float operand) {
      bool ok = false;
      float old_val = 0.0f;
      while (!ok) {
        old_val = *dest;
        float new_val = (old_val < operand) ? old_val : operand;
        ok = atomic_compare_exchange_weak_explicit(
            (device atomic_int *)dest, (thread int *)(&old_val),
            *((thread int *)(&new_val)), metal::memory_order_relaxed,
            metal::memory_order_relaxed);
      }
      return old_val;
    }

    float fatomic_fetch_max(device float *dest, const float operand) {
      bool ok = false;
      float old_val = 0.0f;
      while (!ok) {
        old_val = *dest;
        float new_val = (old_val > operand) ? old_val : operand;
        ok = atomic_compare_exchange_weak_explicit(
            (device atomic_int *)dest, (thread int *)(&old_val),
            *((thread int *)(&new_val)), metal::memory_order_relaxed,
            metal::memory_order_relaxed);
      }
      return old_val;
    }

    float mtl_rounding_prepare_float(float f) {
      // See taichi/runtime/llvm/runtime.cpp
      const int32_t delta_bits =
          (union_cast<int32_t>(f) & 0x80000000) | union_cast<int32_t>(0.5f);
      const float delta = union_cast<float>(delta_bits);
      return f + delta;
    }

    // P for (p)hysical type
    template <typename P>
    void mtl_set_partial_bits(device P *ptr, P value, uint32_t offset,
                              uint32_t bits) {
      // See taichi/runtime/llvm/runtime.cpp
      static_assert(is_same<P, int32_t>::value || is_same<P, uint32_t>::value,
                    "unsupported atomic type");
      constexpr int N = sizeof(P) * 8;
      // precondition: |mask| & |value| == |value|
      const uint32_t mask =
          ((~(uint32_t)0U) << (N - bits)) >> (N - offset - bits);
      device auto *atm_ptr = reinterpret_cast<device _atomic<P> *>(ptr);
      bool ok = false;
      while (!ok) {
        P old_val = *ptr;
        P new_val = (old_val & (~mask)) | (value << offset);
        ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val,
                                                   metal::memory_order_relaxed,
                                                   metal::memory_order_relaxed);
      }
    }

    namespace detail {
      template <bool Signed>
      struct SHRCaster {
        using type = int32_t;
      };

      template <>
      struct SHRCaster<false> {
        using type = uint32_t;
      };
    }  // namespace detail

    // C for (c)ompute type, P for (p)hysical type
    template <typename C, typename P>
    C mtl_get_partial_bits(device P *ptr, uint32_t offset, uint32_t bits) {
      static_assert(is_same<P, int32_t>::value || is_same<P, uint32_t>::value,
                    "unsupported atomic type");
      constexpr int N = sizeof(P) * 8;
      const P phy_val = *ptr;
      using PCasted = typename detail::SHRCaster<is_signed<C>::value>::type;
      // SHL is identical between signed and unsigned integrals.
      const auto step1 = static_cast<PCasted>(phy_val << (N - (offset + bits)));
      // ASHR vs LSHR is implicitly encoded in type TCasted.
      return static_cast<C>(step1 >> (N - bits));
    }

    struct RandState { uint32_t seed; };

    uint32_t metal_rand_u32(device RandState * state) {
      device uint *sp = (device uint *)&(state->seed);
      bool done = false;
      uint32_t nxt = 0;
      while (!done) {
        uint32_t o = *sp;
        // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        nxt = o * 1103515245 + 12345;
        done = atomic_compare_exchange_weak_explicit(
            (device atomic_uint *)sp, &o, nxt, metal::memory_order_relaxed,
            metal::memory_order_relaxed);
      }
      return nxt * 1000000007;
    }

    int32_t metal_rand_i32(device RandState * state) {
      return metal_rand_u32(state);
    }

    float metal_rand_f32(device RandState *state) {
      return metal_rand_u32(state) * (1.0f / 4294967296.0f);
    }
    // clang-format off
)
METAL_END_HELPERS_DEF
// clang-format on

#undef METAL_BEGIN_HELPERS_DEF
#undef METAL_END_HELPERS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
