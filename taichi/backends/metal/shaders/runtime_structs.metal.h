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
    constant constexpr int kTaichiMaxNumIndices = 8;
    constant constexpr int kTaichiNumChunks = 1024;
    constant constexpr int kAlignment = 8;
    using PtrOffset = int32_t;

    struct MemoryAllocator {
      atomic_int next;

      constant constexpr static int kInitOffset = 8;

      static inline bool is_valid(PtrOffset v) {
        return v >= kInitOffset;
      }
    };

    // ListManagerData manages a list of elements with adjustable size.
    struct ListManagerData {
      int32_t element_stride = 0;

      int32_t log2_num_elems_per_chunk = 0;
      // Index to the next element in this list.
      // |next| can never go beyond |kTaichiNumChunks| * |num_elems_per_chunk|.
      atomic_int next;

      atomic_int chunks[kTaichiNumChunks];

      struct ReservedElemPtrOffset {
       public:
        ReservedElemPtrOffset() = default;
        explicit ReservedElemPtrOffset(PtrOffset v) : val_(v) {
        }

        inline bool is_valid() const {
          return is_valid(val_);
        }

        inline static bool is_valid(PtrOffset v) {
          return MemoryAllocator::is_valid(v);
        }

        inline PtrOffset value() const {
          return val_;
        }

       private:
        PtrOffset val_{0};
      };
    };

    // NodeManagerData stores the actual data needed to implement NodeManager
    // in Metal buffers.
    //
    // The actual allocated elements are not embedded in the memory region of
    // NodeManagerData. Instead, all this data structure does is to maintain a
    // few lists (ListManagerData). In particular, |data_list| stores the actual
    // data, while |free_list| and |recycle_list| are only meant for GC.
    struct NodeManagerData {
      using ElemIndex = ListManagerData::ReservedElemPtrOffset;
      // Stores the actual data.
      ListManagerData data_list;
      // For GC
      ListManagerData free_list;
      ListManagerData recycled_list;
      atomic_int free_list_used;
      // Need this field to bookkeep some data during GC
      int recycled_list_size_backup;
    };

    // This class is very similar to metal::SNodeDescriptor
    struct SNodeMeta {
      enum Type {
        Root = 0,
        Dense = 1,
        Bitmasked = 2,
        Dynamic = 3,
        Pointer = 4,
        BitStruct = 5,
      };
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
        int32_t num_elements_from_root = 0;
      };

      Extractor extractors[kTaichiMaxNumIndices];
    };

    struct ElementCoords { int32_t at[kTaichiMaxNumIndices]; };

    struct ListgenElement {
      ElementCoords coords;
      // Memory offset from a given address.
      // * If in_root_buffer() is true, this is from the root buffer address.
      // * O/W this is from the |id|-th NodeManager's |elem_idx|-th element.
      int32_t mem_offset = 0;

      struct BelongedNodeManager {
        // Index of the *NodeManager itself* in the runtime buffer.
        // If -1, the memory where this cell lives isn't in a particular
        // NodeManager's dynamically allocated memory. Instead, it is at a fixed
        // location in the root buffer.
        //
        // For {dense, bitmasked}, this should always be -1.
        int32_t id = -1;
        // Index of the element within the NodeManager.
        NodeManagerData::ElemIndex elem_idx;
      };
      BelongedNodeManager belonged_nodemgr;

      inline bool in_root_buffer() const {
        return belonged_nodemgr.id < 0;
      }
    };
)
METAL_END_RUNTIME_STRUCTS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_STRUCTS_DEF
#undef METAL_END_RUNTIME_STRUCTS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
