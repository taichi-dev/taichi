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
    // The actual allocated elements are not embedded in the memory region of
    // NodeManagerData. Instead, all this data structure does is to maintain a
    // few lists (ListManagerData). In particular, |data_list| stores the actual
    // data, while |free_list| and |recycle_list| are only meant for GC.
    struct NodeManagerData {
      // Stores the actual data.
      ListManagerData data_list;
      // For GC
      ListManagerData free_list;
      ListManagerData recycled_list;
      atomic_int free_list_used;
      // Need this field to bookkeep some data during GC
      int recycled_list_size_backup;

      // Use this type instead of the raw index type (int32_t), because the
      // raw value needs to be shifted by |kIndexOffset| in order for the
      // spinning memory allocation algorithm to work.
      struct ElemIndex {
        // The first 8 index values are reserved to encode special status:
        // * 0  : nullptr
        // * 1  : spinning for allocation
        // * 2-7: unused for now
        //
        /// For each allocated index, it is added by |index_offset| to skip over
        /// these reserved values.
        constant static constexpr int32_t kIndexOffset = 8;

        ElemIndex() = default;

        static ElemIndex from_index(int i) {
          return ElemIndex(i + kIndexOffset);
        }

        static ElemIndex from_raw(int r) {
          return ElemIndex(r);
        }

        inline int32_t index() const {
          return raw_ - kIndexOffset;
        }

        inline int32_t raw() const {
          return raw_;
        }

        inline bool is_valid() const {
          return raw_ >= kIndexOffset;
        }

        inline static bool is_valid(int raw) {
          return ElemIndex::from_raw(raw).is_valid();
        }

       private:
        explicit ElemIndex(int r) : raw_(r) {
        }
        int32_t raw_ = 0;
      };
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

    struct ElementCoords { int32_t at[kTaichiMaxNumIndices]; };

    struct ListgenElement {
      ElementCoords coords;
      // Memory offset from a given address.
      // * If in_root_buffer() is true, this is from the root buffer address.
      // * O.W. this is from the |id|-th NodeManager's |elem_idx|-th element.
      int32_t mem_offset = 0;

      inline bool in_root_buffer() const {
        // Placeholder impl
        return true;
      }
    };
    // clang-format off
)
METAL_END_RUNTIME_STRUCTS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_STRUCTS_DEF
#undef METAL_END_RUNTIME_STRUCTS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
