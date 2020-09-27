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

    // NodeManagerData stores the actual data needed to implement NodeManager
    // in Metal buffers.
    //
    // There are several level of indirections here to retrieve an allocated
    // element from a NodeManager. The actual allocated elements are not
    // embedded in the memory region of NodeManagerData. Instead, all this data
    // structure does is to maintain a few lists (ListManagerData).
    //
    // However, these lists do not store the actual data, either. Instead, their
    // elements are just 32-bit integers, which are memory offsets (PtrOffset)
    // in a Metal buffer. That buffer to which these offsets point holds the
    // actual data.
    struct NodeManagerData {
      using ElemIndex = int32_t;
      // Stores the actual data.
      ListManagerData data_list;
      // For GC
      ListManagerData free_list;
      ListManagerData recycled_list;
      atomic_int free_list_used;
      // Need this field to bookkeep some data during GC
      int recycled_list_size_backup;
      // The first 8 index values are reserved to encode special status:
      // * 0  : nullptr
      // * 1  : spinning for allocation
      // * 2-7: unused for now
      //
      /// For each allocated index, it is added by |index_offset| to skip over
      /// these reserved values.
      constant static constexpr ElemIndex kIndexOffset = 8;
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
