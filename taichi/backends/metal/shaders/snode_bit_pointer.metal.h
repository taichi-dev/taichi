#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_SRC_DEF constexpr auto kMetalSNodeBitPointerSourceCode =
#define METAL_END_SRC_DEF ;
#else
#define METAL_BEGIN_SRC_DEF
#define METAL_END_SRC_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

#define METAL_BEGIN_SRC_DEF
#define METAL_END_SRC_DEF

#include <type_traits>

using std::is_same;
using std::is_signed;

#endif  // TI_INSIDE_METAL_CODEGEN

METAL_BEGIN_SRC_DEF
STR(
    // SNodeBitPointer is used as the value type for bit_struct SNodes on Metal.
    struct SNodeBitPointer {
      // Physical type is hardcoded to uint32_t. This is a restriction because
      // Metal only supports 32-bit int/uint atomics.
      device uint32_t *base;
      uint32_t offset;

      SNodeBitPointer(device byte * b, uint32_t o)
          : base((device uint32_t *)b), offset(o) {
      }
    };

    // |f| should already be scaled. |C| is the compute type.
    template <typename C>
    C mtl_quant_fixed_to_quant_int(float f) {
      // Branch free implementation of `f + sign(f) * 0.5`.
      // See rounding_prepare_f* in taichi/runtime/llvm/runtime.cpp
      const int32_t delta_bits =
          (union_cast<int32_t>(f) & 0x80000000) | union_cast<int32_t>(0.5f);
      const float delta = union_cast<float>(delta_bits);
      return static_cast<C>(f + delta);
    }

    void mtl_set_partial_bits(SNodeBitPointer bp,
                              uint32_t value,
                              uint32_t bits) {
      // See taichi/runtime/llvm/runtime.cpp
      //
      // We could have encoded |bits| as a compile time constant, but I guess
      // the performance improvement is negligible.
      using P = uint32_t;  // (P)hysical type
      constexpr int N = sizeof(P) * 8;
      // precondition: |mask| & |value| == |value|
      const uint32_t mask =
          ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits);
      device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base);
      bool ok = false;
      while (!ok) {
        P old_val = *(bp.base);
        P new_val = (old_val & (~mask)) | (value << bp.offset);
        ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val,
                                                   metal::memory_order_relaxed,
                                                   metal::memory_order_relaxed);
      }
    }

    void mtl_set_full_bits(SNodeBitPointer bp, uint32_t value) {
      device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base);
      atomic_store_explicit(atm_ptr, value, metal::memory_order_relaxed);
    }

    uint32_t mtl_atomic_add_partial_bits(SNodeBitPointer bp,
                                         uint32_t value,
                                         uint32_t bits) {
      // See taichi/runtime/llvm/runtime.cpp
      using P = uint32_t;  // (P)hysical type
      constexpr int N = sizeof(P) * 8;
      // precondition: |mask| & |value| == |value|
      const uint32_t mask =
          ((~(uint32_t)0U) << (N - bits)) >> (N - bp.offset - bits);
      device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base);
      P old_val = 0;
      bool ok = false;
      while (!ok) {
        old_val = *(bp.base);
        P new_val = old_val + (value << bp.offset);
        // The above computation might overflow |bits|, so we have to OR them
        // again, with the mask applied.
        new_val = (old_val & (~mask)) | (new_val & mask);
        ok = atomic_compare_exchange_weak_explicit(atm_ptr, &old_val, new_val,
                                                   metal::memory_order_relaxed,
                                                   metal::memory_order_relaxed);
      }
      return old_val;
    }

    uint32_t mtl_atomic_add_full_bits(SNodeBitPointer bp, uint32_t value) {
      // When all the bits are used, we can replace CAS with a simple add.
      device auto *atm_ptr = reinterpret_cast<device atomic_uint *>(bp.base);
      return atomic_fetch_add_explicit(atm_ptr, value,
                                       metal::memory_order_relaxed);
    }

    namespace detail {
      // Metal supports C++ template specialization... what a crazy world
      template <bool Signed>
      struct SHRSelector {
        using type = int32_t;
      };

      template <>
      struct SHRSelector<false> {
        using type = uint32_t;
      };
    }  // namespace detail

    // (C)ompute type
    template <typename C>
    C mtl_get_partial_bits(SNodeBitPointer bp, uint32_t bits) {
      using P = uint32_t;  // (P)hysical type
      constexpr int N = sizeof(P) * 8;
      const P phy_val = *(bp.base);
      // Use CSel instead of C to preserve the bit width.
      using CSel = typename detail::SHRSelector<is_signed<C>::value>::type;
      // SHL is identical between signed and unsigned integrals.
      const auto step1 = static_cast<CSel>(phy_val << (N - (bp.offset + bits)));
      // ASHR vs LSHR is implicitly encoded in type CSel.
      return static_cast<C>(step1 >> (N - bits));
    }

    template <typename C>
    C mtl_get_full_bits(SNodeBitPointer bp) {
      return static_cast<C>(*(bp.base));
    })
METAL_END_SRC_DEF
// clang-format on

#undef METAL_BEGIN_SRC_DEF
#undef METAL_END_SRC_DEF

#include "taichi/backends/metal/shaders/epilog.h"
