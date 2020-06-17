#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_AD_STACK_DEF constexpr auto kMetalAdStackSourceCode =
#define METAL_END_AD_STACK_DEF ;
#else
#define METAL_BEGIN_AD_STACK_DEF
#define METAL_END_AD_STACK_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

#include <cstdint>

#define METAL_BEGIN_AD_STACK_DEF
#define METAL_END_AD_STACK_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

// Autodiff stack for local mutables

// clang-format off
METAL_BEGIN_AD_STACK_DEF
STR(
    // clang-format on
    using AdStackPtr = thread byte *;

    inline thread uint32_t *
    mtl_ad_stack_n(AdStackPtr stack) {
      return reinterpret_cast<thread uint32_t *>(stack);
    }

    inline AdStackPtr mtl_ad_stack_data(AdStackPtr stack) {
      return stack + sizeof(uint32_t);
    }

    inline void mtl_ad_stack_init(AdStackPtr stack) {
      *mtl_ad_stack_n(stack) = 0;
    }

    inline AdStackPtr mtl_ad_stack_top_primal(AdStackPtr stack,
                                              int element_size) {
      const auto n = *mtl_ad_stack_n(stack);
      return mtl_ad_stack_data(stack) + (n - 1) * 2 * element_size;
    }

    inline AdStackPtr mtl_ad_stack_top_adjoint(AdStackPtr stack,
                                               int element_size) {
      return mtl_ad_stack_top_primal(stack, element_size) + element_size;
    }

    inline void mtl_ad_stack_pop(AdStackPtr stack) {
      thread auto &n = *mtl_ad_stack_n(stack);
      --n;
    }

    void mtl_ad_stack_push(AdStackPtr stack, int element_size) {
      thread auto &n = *mtl_ad_stack_n(stack);
      ++n;

      AdStackPtr data = mtl_ad_stack_top_primal(stack, element_size);
      for (int i = 0; i < element_size * 2; ++i) {
        data[i] = 0;
      }
    }
    // clang-format off
)
METAL_END_AD_STACK_DEF
// clang-format on

#undef METAL_BEGIN_AD_STACK_DEF
#undef METAL_END_AD_STACK_DEF

#include "taichi/backends/metal/shaders/epilog.h"
