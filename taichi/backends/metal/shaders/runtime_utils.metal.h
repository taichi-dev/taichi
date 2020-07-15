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

    [[maybe_unused]] PtrOffset mtl_memalloc_alloc(device MemoryAllocator *ma,
                                                  int32_t size) {
      size = ((size + kAlignment - 1) / kAlignment) * kAlignment;
      return atomic_fetch_add_explicit(&ma->next, size,
                                       metal::memory_order_relaxed);
    }

    [[maybe_unused]] device char
        *mtl_memalloc_to_ptr(device MemoryAllocator *ma, PtrOffset offs) {
          return reinterpret_cast<device char *>(ma + 1) + offs;
        }

    struct ListManager {
      device ListManagerData *lm_data;
      device MemoryAllocator *mem_alloc;

      inline int num_active() {
        return atomic_load_explicit(&(lm_data->next),
                                    metal::memory_order_relaxed);
      }

      inline void resize(int sz) {
        atomic_store_explicit(&(lm_data->next), sz,
                              metal::memory_order_relaxed);
      }

      inline void clear() {
        resize(0);
      }

      struct ReserveElemResult {
        int elem_idx;
        PtrOffset chunk_ptr_offs;
      };

      ReserveElemResult reserve_new_elem() {
        const int elem_idx = atomic_fetch_add_explicit(
            &lm_data->next, 1, metal::memory_order_relaxed);
        const int chunk_idx = elem_idx >> lm_data->log2_num_elems_per_chunk;
        const PtrOffset chunk_ptr_offs = ensure_chunk(chunk_idx);
        return {elem_idx, chunk_ptr_offs};
      }

      device char *append() {
        auto reserved = reserve_new_elem();
        return get_elem_from_chunk(reserved.elem_idx, reserved.chunk_ptr_offs);
      }

      template <typename T>
      void append(thread const T &elem) {
        device char *ptr = append();
        thread char *elem_ptr = (thread char *)(&elem);

        for (int i = 0; i < lm_data->element_stride; ++i) {
          *ptr = *elem_ptr;
          ++ptr;
          ++elem_ptr;
        }
      }

      device char *get_ptr(int i) {
        const int chunk_idx = i >> lm_data->log2_num_elems_per_chunk;
        const PtrOffset chunk_ptr_offs = atomic_load_explicit(
            lm_data->chunks + chunk_idx, metal::memory_order_relaxed);
        return get_elem_from_chunk(i, chunk_ptr_offs);
      }

      template <typename T>
      T get(int i) {
        return *reinterpret_cast<device T *>(get_ptr(i));
      }

     private:
      PtrOffset ensure_chunk(int i) {
        PtrOffset offs = 0;
        const int chunk_bytes =
            (lm_data->element_stride << lm_data->log2_num_elems_per_chunk);

        while (true) {
          int stored = 0;
          // If chunks[i] is unallocated, i.e. 0, mark it as 1 to prevent others
          // from requesting memory again. Once allocated, set chunks[i] to the
          // actual address offset, which is guaranteed to be greater than 1.
          const bool is_me = atomic_compare_exchange_weak_explicit(
              lm_data->chunks + i, &stored, 1, metal::memory_order_relaxed,
              metal::memory_order_relaxed);
          if (is_me) {
            offs = mtl_memalloc_alloc(mem_alloc, chunk_bytes);
            atomic_store_explicit(lm_data->chunks + i, offs,
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

      device char *get_elem_from_chunk(int i, PtrOffset chunk_ptr_offs) {
        device char *chunk_ptr = reinterpret_cast<device char *>(
            mtl_memalloc_to_ptr(mem_alloc, chunk_ptr_offs));
        const uint32_t mask = ((1 << lm_data->log2_num_elems_per_chunk) - 1);
        return chunk_ptr + ((i & mask) * lm_data->element_stride);
      }
    };

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
        atomic_fetch_max_explicit(ptr, (uint32_t)(i + 1),
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
    })
METAL_END_RUNTIME_UTILS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_UTILS_DEF
#undef METAL_END_RUNTIME_UTILS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
