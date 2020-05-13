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
        if (n & 1)
          ans *= tmp;
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
