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
    [[maybe_unused]] int num_active(device const ListManager *list) {
      return list->next;
    }

    template <typename T>
    int append(device ListManager *list,
               thread const T &elem,
               device byte *data_addr) {
      thread char *elem_ptr = (thread char *)(&elem);
      int me = atomic_fetch_add_explicit(
          reinterpret_cast<device atomic_int *>(&(list->next)), 1,
          metal::memory_order_relaxed);
      device byte *ptr =
          data_addr + list->mem_begin + (me * list->element_stride);
      for (int i = 0; i < list->element_stride; ++i) {
        *ptr = *elem_ptr;
        ++ptr;
        ++elem_ptr;
      }
      return me;
    }

    template <typename T>
    T get(const device ListManager *list, int i, device const byte *data_addr) {
      return *reinterpret_cast<device const T *>(data_addr + list->mem_begin +
                                                 (i * list->element_stride));
    }

    [[maybe_unused]] void clear(device ListManager *list) {
      atomic_store_explicit(
          reinterpret_cast<device atomic_int *>(&(list->next)), 0,
          metal::memory_order_relaxed);
    }

    [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return true;
      }
      device auto *ptr =
          reinterpret_cast<device atomic_uint *>(
              addr + ((meta.num_slots - i) * meta.element_stride)) +
          (i / (sizeof(uint32_t) * 8));
      uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed);
      return ((bits >> (i % (sizeof(uint32_t) * 8))) & 1);
    }

    [[maybe_unused]] void activate(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return;
      }
      device auto *ptr =
          reinterpret_cast<device atomic_uint *>(
              addr + ((meta.num_slots - i) * meta.element_stride)) +
          (i / (sizeof(uint32_t) * 8));
      const uint32_t mask = (1 << (i % (sizeof(uint32_t) * 8)));
      atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed);
    }

    [[maybe_unused]] void deactivate(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return;
      }
      device auto *ptr =
          reinterpret_cast<device atomic_uint *>(
              addr + ((meta.num_slots - i) * meta.element_stride)) +
          (i / (sizeof(uint32_t) * 8));
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
)
METAL_END_RUNTIME_UTILS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_UTILS_DEF
#undef METAL_END_RUNTIME_UTILS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
