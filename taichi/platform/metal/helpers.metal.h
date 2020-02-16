#ifdef TI_INSIDE_METAL_CODEGEN

#define METAL_BEGIN_HELPERS_DEF constexpr auto kMetalHelpersSourceCode =
#define METAL_END_HELPERS_DEF ;

#define STR2(...) #__VA_ARGS__
#define STR(...) STR2(__VA_ARGS__)

#else

#define METAL_BEGIN_HELPERS_DEF
#define METAL_END_HELPERS_DEF
#define STR(...) __VA_ARGS__

#define device
#define constant
#define thread

using atomic_int = int;

template <typename... Args>
bool atomic_compare_exchange_weak_explicit(Args...) {
  static_assert(false, "Do not include");
}

namespace metal {
bool memory_order_relaxed = false;
}  // namespace metal

#endif  // TI_INSIDE_METAL_CODEGEN

METAL_BEGIN_HELPERS_DEF
STR(
    template <typename T, typename G>
    T union_cast(G g) {
      // For some reason, if I emit taichi/common.h's union_cast(), Metal failed
      // to compile. More strangely, if I copy the generated code to XCode as a
      // Metal kernel, it compiled successfully...
      static_assert(sizeof(T) == sizeof(G), "Size mismatch");
      return *reinterpret_cast<thread const T *>(&g);
    }

    inline int ifloordiv(int lhs, int rhs) {
      const int intm = (lhs / rhs);
      return (((lhs * rhs < 0) && (rhs * intm != lhs)) ? (intm - 1) : intm);
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
    })
METAL_END_HELPERS_DEF

#undef METAL_BEGIN_HELPERS_DEF
#undef METAL_END_HELPERS_DEF
#undef STR2
#undef STR
