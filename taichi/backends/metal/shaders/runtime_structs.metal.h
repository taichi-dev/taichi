#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_RUNTIME_STRUCTS_DEF \
  constexpr auto kMetalRuntimeStructsSourceCode =
#define METAL_END_RUNTIME_STRUCTS_DEF ;
#else
#define METAL_BEGIN_RUNTIME_STRUCTS_DEF
#define METAL_END_RUNTIME_STRUCTS_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

#include <cstdint>

#include "taichi/inc/constants.h"

static_assert(taichi_max_num_indices == 8,
              "Please update kTaichiMaxNumIndices");
static_assert(sizeof(char *) == 8, "Metal pointers are 64-bit.");
#define METAL_BEGIN_RUNTIME_STRUCTS_DEF
#define METAL_END_RUNTIME_STRUCTS_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

// clang-format off
METAL_BEGIN_RUNTIME_STRUCTS_DEF
STR(
    // clang-format on
    constant constexpr int kTaichiMaxNumIndices = 8;
    constant constexpr int kTaichiNumChunks = 1024;

    struct MemoryAllocator { atomic_int next; };

    struct ListgenElement {
      int32_t coords[kTaichiMaxNumIndices];
      int32_t root_mem_offset = 0;
    };

    // ListManagerData manages a list of elements with adjustable size.
    struct ListManagerData {
      int32_t element_stride = 0;

      int32_t log2_num_elems_per_chunk = 0;
      // Index to the next element in this list.
      // |next| can never go beyond |kTaichiNumChunks| * |num_elems_per_chunk|.
      atomic_int next;

      atomic_int chunks[kTaichiNumChunks];
    };

    // This class is very similar to metal::SNodeDescriptor
    struct SNodeMeta {
      enum Type { Root = 0, Dense = 1, Bitmasked = 2, Dynamic = 3 };
      int32_t element_stride = 0;
      int32_t num_slots = 0;
      int32_t mem_offset_in_parent = 0;
      int32_t type = 0;
    };

    struct SNodeExtractors {
      struct Extractor {
        int32_t start = 0;
        int32_t num_bits = 0;
        int32_t acc_offset = 0;
        int32_t num_elements = 0;
      };

      Extractor extractors[kTaichiMaxNumIndices];
    };
    // clang-format off
)
METAL_END_RUNTIME_STRUCTS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_STRUCTS_DEF
#undef METAL_END_RUNTIME_STRUCTS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
