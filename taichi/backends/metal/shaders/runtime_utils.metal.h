#ifndef TI_METAL_NESTED_INCLUDE

#define TI_METAL_NESTED_INCLUDE
#include "taichi/backends/metal/shaders/runtime_structs.metal.h"
#undef TI_METAL_NESTED_INCLUDE

#else
#include "taichi/backends/metal/shaders/runtime_structs.metal.h"
#endif  // TI_METAL_NESTED_INCLUDE

#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_RUNTIME_UTILS_DEF \
  constexpr auto kMetalRuntimeUtilsSourceCode =
#define METAL_END_RUNTIME_UTILS_DEF ;
#else
#define METAL_BEGIN_RUNTIME_UTILS_DEF
#define METAL_END_RUNTIME_UTILS_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

#define METAL_BEGIN_RUNTIME_UTILS_DEF
#define METAL_END_RUNTIME_UTILS_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

// clang-format off
METAL_BEGIN_RUNTIME_UTILS_DEF
STR(
    using PtrOffset = int32_t;
    constant constexpr int kAlignment = 8;

    [[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator * ma,
                                                  int32_t size) {
      size = ((size + kAlignment - 1) / kAlignment) * kAlignment;
      return atomic_fetch_add_explicit(&ma->next, size,
                                       metal::memory_order_relaxed);
    }

    [[maybe_unused]] device char
        *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) {
          return reinterpret_cast<device char *>(ma + 1) + offs;
        }

    [[maybe_unused]] int num_active(thread ListManager *l) {
      return atomic_load_explicit(&(l->lm_data->next),
                                  metal::memory_order_relaxed);
    }

    [[maybe_unused]] void clear(thread ListManager *l) {
      atomic_store_explicit(&(l->lm_data->next), 0,
                            metal::memory_order_relaxed);
    }

    [[maybe_unused]] PtrOffset mtl_listmgr_ensure_chunk(thread ListManager *l,
                                                        int i) {
      device ListManagerData *list = l->lm_data;
      PtrOffset offs = 0;
      const int kChunkBytes =
          (list->element_stride << list->log2_num_elems_per_chunk);

      while (true) {
        int stored = 0;
        // If chunks[i] is unallocated, i.e. 0, mark it as 1 to prevent others
        // from requesting memory again. Once allocated, set chunks[i] to the
        // actual address offset, which is guaranteed to be greater than 1.
        const bool is_me = atomic_compare_exchange_weak_explicit(
            list->chunks + i, &stored, 1, metal::memory_order_relaxed,
            metal::memory_order_relaxed);
        if (is_me) {
          offs = mtl_memalloc_alloc(l->mem_alloc, kChunkBytes);
          atomic_store_explicit(list->chunks + i, offs,
                                metal::memory_order_relaxed);
          break;
        } else if (stored > 1) {
          offs = stored;
          break;
        }
        // |stored| == 1, just spin
      }
      return offs;
    }

    [[maybe_unused]] device char *mtl_listmgr_get_elem_from_chunk(
        thread ListManager *l,
        int i,
        PtrOffset chunk_ptr_offs) {
      device ListManagerData *list = l->lm_data;
      device char *chunk_ptr = reinterpret_cast<device char *>(
          mtl_memalloc_to_ptr(l->mem_alloc, chunk_ptr_offs));
      const uint32_t mask = ((1 << list->log2_num_elems_per_chunk) - 1);
      return chunk_ptr + ((i & mask) * list->element_stride);
    }

    [[maybe_unused]] device char *append(thread ListManager *l) {
      device ListManagerData *list = l->lm_data;
      const int elem_idx = atomic_fetch_add_explicit(
          &list->next, 1, metal::memory_order_relaxed);
      const int chunk_idx = elem_idx >> list->log2_num_elems_per_chunk;
      const PtrOffset chunk_ptr_offs = mtl_listmgr_ensure_chunk(l, chunk_idx);
      return mtl_listmgr_get_elem_from_chunk(l, elem_idx, chunk_ptr_offs);
    }

    template <typename T>
    [[maybe_unused]] void append(thread ListManager *l, thread const T &elem) {
      device char *ptr = append(l);
      thread char *elem_ptr = (thread char *)(&elem);

      for (int i = 0; i < l->lm_data->element_stride; ++i) {
        *ptr = *elem_ptr;
        ++ptr;
        ++elem_ptr;
      }
    }

    template <typename T>
    [[maybe_unused]] T get(thread ListManager *l, int i) {
      device ListManagerData *list = l->lm_data;
      const int chunk_idx = i >> list->log2_num_elems_per_chunk;
      const PtrOffset chunk_ptr_offs = atomic_load_explicit(
          list->chunks + chunk_idx, metal::memory_order_relaxed);
      return *reinterpret_cast<device T *>(
          mtl_listmgr_get_elem_from_chunk(l, i, chunk_ptr_offs));
    }

    [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return true;
      }
      device auto *meta_ptr_begin = reinterpret_cast<device atomic_uint *>(
          addr + ((meta.num_slots - i) * meta.element_stride));
      if (meta.type == SNodeMeta::Dynamic) {
        device auto *ptr = meta_ptr_begin;
        uint32_t n = atomic_load_explicit(ptr, metal::memory_order_relaxed);
        return i < n;
      }
      device auto *ptr = meta_ptr_begin + (i / (sizeof(uint32_t) * 8));
      uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed);
      return ((bits >> (i % (sizeof(uint32_t) * 8))) & 1);
    }

    [[maybe_unused]] void activate(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return;
      }
      device auto *meta_ptr_begin = reinterpret_cast<device atomic_uint *>(
          addr + ((meta.num_slots - i) * meta.element_stride));
      if (meta.type == SNodeMeta::Dynamic) {
        device auto *ptr = meta_ptr_begin;
        // Unfortunately we cannot check if i + 1 is in bound
        atomic_store_explicit(ptr, (uint32_t)(i + 1),
                              metal::memory_order_relaxed);
        return;
      }
      device auto *ptr = meta_ptr_begin + (i / (sizeof(uint32_t) * 8));
      const uint32_t mask = (1 << (i % (sizeof(uint32_t) * 8)));
      atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed);
    }

    [[maybe_unused]] void deactivate(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return;
      }
      device auto *meta_ptr_begin = reinterpret_cast<device atomic_uint *>(
          addr + ((meta.num_slots - i) * meta.element_stride));
      if (meta.type == SNodeMeta::Dynamic) {
        device auto *ptr = meta_ptr_begin;
        // For dynamic, deactivate() applies for all the slots
        atomic_store_explicit(ptr, 0u, metal::memory_order_relaxed);
        return;
      }
      device auto *ptr = meta_ptr_begin + (i / (sizeof(uint32_t) * 8));
      const uint32_t mask = ~(1 << (i % (sizeof(uint32_t) * 8)));
      atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed);
    }

    [[maybe_unused]] void refine_coordinates(
        thread const ListgenElement &parent_elem,
        device const SNodeExtractors &child_extrators,
        int l,
        thread ListgenElement *child_elem) {
      for (int i = 0; i < kTaichiMaxNumIndices; ++i) {
        device const auto &ex = child_extrators.extractors[i];
        const int mask = ((1 << ex.num_bits) - 1);
        const int addition = (((l >> ex.acc_offset) & mask) << ex.start);
        child_elem->coords[i] = (parent_elem.coords[i] | addition);
      }
    }

    [[maybe_unused]] int dynamic_append(device byte *addr,
                                        SNodeMeta meta,
                                        int32_t data) {
      // |addr| always starts at the beginning of the dynamic
      device auto *n_ptr = reinterpret_cast<device atomic_int *>(
          addr + (meta.num_slots * meta.element_stride));
      int me = atomic_fetch_add_explicit(n_ptr, 1, metal::memory_order_relaxed);
      *(reinterpret_cast<device int32_t *>(addr) + me) = data;
      return me;
    }

    [[maybe_unused]] int dynamic_length(device byte *addr, SNodeMeta meta) {
      // |addr| always starts at the beginning of the dynamic
      device auto *n_ptr = reinterpret_cast<device atomic_int *>(
          addr + (meta.num_slots * meta.element_stride));
      return atomic_load_explicit(n_ptr, metal::memory_order_relaxed);
    }
)
METAL_END_RUNTIME_UTILS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_UTILS_DEF
#undef METAL_END_RUNTIME_UTILS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
