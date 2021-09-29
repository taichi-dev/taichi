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

// Just a mock to illustrate what the Runtime looks like, do not use.
// The actual Runtime struct has to be emitted by codegen, because it depends
// on the number of SNodes.
struct Runtime {
  SNodeMeta *snode_metas = nullptr;
  SNodeExtractors *snode_extractors = nullptr;
  ListManagerData *snode_lists = nullptr;
  NodeManagerData *snode_allocators = nullptr;
  NodeManagerData::ElemIndex *ambient_indices = nullptr;
  uint32_t *rand_seeds = nullptr;
};

#define METAL_BEGIN_RUNTIME_UTILS_DEF
#define METAL_END_RUNTIME_UTILS_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

METAL_BEGIN_RUNTIME_UTILS_DEF
STR(
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
      using ReservedElemPtrOffset = ListManagerData::ReservedElemPtrOffset;
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

      ReservedElemPtrOffset reserve_new_elem() {
        const int elem_idx = atomic_fetch_add_explicit(
            &lm_data->next, 1, metal::memory_order_relaxed);
        const int chunk_idx = get_chunk_index(elem_idx);
        const PtrOffset chunk_ptr_offs = ensure_chunk(chunk_idx);
        const auto offset =
            get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs);
        return ReservedElemPtrOffset{offset};
      }

      device char *append() {
        auto reserved = reserve_new_elem();
        return get_ptr(reserved);
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

      device char *get_ptr(ReservedElemPtrOffset offs) {
        return mtl_memalloc_to_ptr(mem_alloc, offs.value());
      }

      device char *get_ptr(int i) {
        const int chunk_idx = get_chunk_index(i);
        const PtrOffset chunk_ptr_offs = atomic_load_explicit(
            lm_data->chunks + chunk_idx, metal::memory_order_relaxed);
        return get_elem_from_chunk(i, chunk_ptr_offs);
      }

      template <typename T>
      T get(int i) {
        return *reinterpret_cast<device T *>(get_ptr(i));
      }

     private:
      inline int get_chunk_index(int elem_idx) const {
        return elem_idx >> lm_data->log2_num_elems_per_chunk;
      }

      PtrOffset ensure_chunk(int chunk_idx) {
        PtrOffset offs = 0;
        const int chunk_bytes =
            (lm_data->element_stride << lm_data->log2_num_elems_per_chunk);

        while (true) {
          int stored = 0;
          // If chunks[i] is unallocated, i.e. 0, mark it as 1 to prevent others
          // from requesting memory again. Once allocated, set chunks[i] to the
          // actual address offset, which is guaranteed to be greater than 1.
          const bool is_me = atomic_compare_exchange_weak_explicit(
              lm_data->chunks + chunk_idx, &stored, 1,
              metal::memory_order_relaxed, metal::memory_order_relaxed);
          if (is_me) {
            offs = mtl_memalloc_alloc(mem_alloc, chunk_bytes);
            atomic_store_explicit(lm_data->chunks + chunk_idx, offs,
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

      PtrOffset get_elem_ptr_offs_from_chunk(int elem_idx,
                                             PtrOffset chunk_ptr_offs) {
        const uint32_t mask = ((1 << lm_data->log2_num_elems_per_chunk) - 1);
        return chunk_ptr_offs + ((elem_idx & mask) * lm_data->element_stride);
      }

      device char *get_elem_from_chunk(int elem_idx, PtrOffset chunk_ptr_offs) {
        const auto offs =
            get_elem_ptr_offs_from_chunk(elem_idx, chunk_ptr_offs);
        return mtl_memalloc_to_ptr(mem_alloc, offs);
      }
    };

    // NodeManager is a GPU-side memory allocator with GC support.
    struct NodeManager {
      using ElemIndex = NodeManagerData::ElemIndex;
      device NodeManagerData *nm_data;
      device MemoryAllocator *mem_alloc;

      ElemIndex allocate() {
        ListManager free_list;
        free_list.lm_data = &(nm_data->free_list);
        free_list.mem_alloc = mem_alloc;
        ListManager data_list;
        data_list.lm_data = &(nm_data->data_list);
        data_list.mem_alloc = mem_alloc;

        const int cur_used = atomic_fetch_add_explicit(
            &(nm_data->free_list_used), 1, metal::memory_order_relaxed);
        if (cur_used < free_list.num_active()) {
          return free_list.get<ElemIndex>(cur_used);
        }

        return data_list.reserve_new_elem();
      }

      device byte *get(ElemIndex i) {
        ListManager data_list;
        data_list.lm_data = &(nm_data->data_list);
        data_list.mem_alloc = mem_alloc;

        return data_list.get_ptr(i);
      }

      void recycle(ElemIndex i) {
        ListManager recycled_list;
        recycled_list.lm_data = &(nm_data->recycled_list);
        recycled_list.mem_alloc = mem_alloc;
        recycled_list.append(i);
      }
    };

    // To make codegen implementation easier, I've made these exceptions:
    // * The somewhat strange SNodeRep_* naming style.
    // * init(), instead of doing initiliaztion in the constructor.
    class SNodeRep_dense {
     public:
      void init(device byte * addr) {
        addr_ = addr;
      }

      inline device byte *addr() {
        return addr_;
      }

      inline bool is_active(int) {
        return true;
      }

      inline void activate(int) {
      }

      inline void deactivate(int) {
      }

     private:
      device byte *addr_ = nullptr;
    };

    using SNodeRep_root = SNodeRep_dense;

    class SNodeRep_bitmasked {
     public:
      constant static constexpr int kBitsPerMask = (sizeof(uint32_t) * 8);

      void init(device byte * addr, int meta_offset) {
        addr_ = addr;
        meta_offset_ = meta_offset;
      }

      inline device byte *addr() {
        return addr_;
      }

      bool is_active(int i) {
        device auto *ptr = to_bitmask_ptr(i);
        uint32_t bits = atomic_load_explicit(ptr, metal::memory_order_relaxed);
        return ((bits >> (i % kBitsPerMask)) & 1);
      }

      void activate(int i) {
        device auto *ptr = to_bitmask_ptr(i);
        const uint32_t mask = (1 << (i % kBitsPerMask));
        atomic_fetch_or_explicit(ptr, mask, metal::memory_order_relaxed);
      }

      void deactivate(int i) {
        device auto *ptr = to_bitmask_ptr(i);
        const uint32_t mask = ~(1 << (i % kBitsPerMask));
        atomic_fetch_and_explicit(ptr, mask, metal::memory_order_relaxed);
      }

     private:
      inline device atomic_uint *to_bitmask_ptr(int i) {
        return reinterpret_cast<device atomic_uint *>(addr_ + meta_offset_) +
               (i / kBitsPerMask);
      }

      device byte *addr_ = nullptr;
      int32_t meta_offset_ = 0;
    };

    class SNodeRep_dynamic {
     public:
      void init(device byte * addr, int meta_offset) {
        addr_ = addr;
        meta_offset_ = meta_offset;
      }

      inline device byte *addr() {
        return addr_;
      }

      bool is_active(int i) {
        const auto n =
            atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed);
        return i < n;
      }

      void activate(int i) {
        device auto *ptr = to_meta_ptr();
        // Unfortunately we cannot check if i + 1 is in bound
        atomic_fetch_max_explicit(ptr, (i + 1), metal::memory_order_relaxed);
        return;
      }

      void deactivate() {
        device auto *ptr = to_meta_ptr();
        // For dynamic, deactivate() applies to all the slots
        atomic_store_explicit(ptr, 0, metal::memory_order_relaxed);
      }

      int append(int32_t data) {
        device auto *ptr = to_meta_ptr();
        // Unfortunately we cannot check if |me| is in bound
        int me = atomic_fetch_add_explicit(ptr, 1, metal::memory_order_relaxed);
        *(reinterpret_cast<device int32_t *>(addr_) + me) = data;
        return me;
      }

      int length() {
        return atomic_load_explicit(to_meta_ptr(), metal::memory_order_relaxed);
      }

     private:
      inline device atomic_int *to_meta_ptr() {
        return reinterpret_cast<device atomic_int *>(addr_ + meta_offset_);
      }

      device byte *addr_ = nullptr;
      int32_t meta_offset_ = 0;
    };

    class SNodeRep_pointer {
     public:
      using ElemIndex = NodeManagerData::ElemIndex;

      void init(device byte * addr, NodeManager nm, ElemIndex ambient_idx) {
        addr_ = addr;
        nm_ = nm;
        ambient_idx_ = ambient_idx;
      }

      device byte *child_or_ambient_addr(int i) {
        auto nm_idx = to_nodemgr_idx(addr_, i);
        nm_idx = nm_idx.is_valid() ? nm_idx : ambient_idx_;
        return nm_.get(nm_idx);
      }

      inline bool is_active(int i) {
        return is_active(addr_, i);
      }

      void activate(int i) {
        device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i);
        auto nm_idx_val =
            atomic_load_explicit(nm_idx_ptr, metal::memory_order_relaxed);
        while (!ElemIndex::is_valid(nm_idx_val)) {
          nm_idx_val = 0;
          // See ListManager::ensure_chunk() for the allocation algorithm.
          // See also https://github.com/taichi-dev/taichi/issues/1174.
          const bool is_me = atomic_compare_exchange_weak_explicit(
              nm_idx_ptr, &nm_idx_val, 1, metal::memory_order_relaxed,
              metal::memory_order_relaxed);
          if (is_me) {
            nm_idx_val = nm_.allocate().value();
            atomic_store_explicit(nm_idx_ptr, nm_idx_val,
                                  metal::memory_order_relaxed);
            break;
          } else if (ElemIndex::is_valid(nm_idx_val)) {
            break;
          }
          // |nm_idx_val| == 1, just spin
        }
      }

      void deactivate(int i) {
        device auto *nm_idx_ptr = to_nodemgr_idx_ptr(addr_, i);
        const auto old_nm_idx_val = atomic_exchange_explicit(
            nm_idx_ptr, 0, metal::memory_order_relaxed);
        const auto old_nm_idx = ElemIndex(old_nm_idx_val);
        if (!old_nm_idx.is_valid()) {
          return;
        }
        nm_.recycle(old_nm_idx);
      }

      static inline device atomic_int *to_nodemgr_idx_ptr(device byte * addr,
                                                          int ch_i) {
        return reinterpret_cast<device atomic_int *>(addr +
                                                     ch_i * sizeof(ElemIndex));
      }

      static inline ElemIndex to_nodemgr_idx(device byte * addr, int ch_i) {
        device auto *ptr = to_nodemgr_idx_ptr(addr, ch_i);
        const auto v = atomic_load_explicit(ptr, metal::memory_order_relaxed);
        return ElemIndex(v);
      }

      static bool is_active(device byte * addr, int ch_i) {
        return to_nodemgr_idx(addr, ch_i).is_valid();
      }

     private:
      device byte *addr_;
      NodeManager nm_;
      // Index of the ambient child element in |nm_|.
      ElemIndex ambient_idx_;
    };

    // This is still necessary in listgen and struct-for kernels, where we don't
    // have the actual SNode structs.
    [[maybe_unused]] int is_active(device byte *addr, SNodeMeta meta, int i) {
      if (meta.type == SNodeMeta::Root || meta.type == SNodeMeta::Dense) {
        return true;
      } else if (meta.type == SNodeMeta::Dynamic) {
        SNodeRep_dynamic rep;
        rep.init(addr, /*meta_offset=*/meta.num_slots * meta.element_stride);
        return rep.is_active(i);
      } else if (meta.type == SNodeMeta::Bitmasked) {
        SNodeRep_bitmasked rep;
        rep.init(addr, /*meta_offset=*/meta.num_slots * meta.element_stride);
        return rep.is_active(i);
      } else if (meta.type == SNodeMeta::Pointer) {
        return SNodeRep_pointer::is_active(addr, i);
      }
      return false;
    }

    [[maybe_unused]] void refine_coordinates(
        thread const ElementCoords &parent,
        device const SNodeExtractors &child_extrators,
        int l,
        thread ElementCoords *child) {
      for (int i = 0; i < kTaichiMaxNumIndices; ++i) {
        device const auto &ex = child_extrators.extractors[i];
        const int mask = ((1 << ex.num_bits) - 1);
        const int addition = ((l >> ex.acc_offset) & mask);
        child->at[i] = ((parent.at[i] << ex.num_bits) | addition);
      }
    }

    // Gets the address of an SNode cell identified by |lgen|.
    [[maybe_unused]] device byte *mtl_lgen_snode_addr(
        thread const ListgenElement &lgen,
        device byte *root_addr,
        device Runtime *rtm,
        device MemoryAllocator *mem_alloc) {
      if (lgen.in_root_buffer()) {
        return root_addr + lgen.mem_offset;
      }
      NodeManager nm;
      nm.nm_data = (rtm->snode_allocators + lgen.belonged_nodemgr.id);
      nm.mem_alloc = mem_alloc;
      device byte *addr = nm.get(lgen.belonged_nodemgr.elem_idx);
      return addr + lgen.mem_offset;
    }

    // GC utils
    [[maybe_unused]] void run_gc_compact_free_list(
        device NodeManagerData *nm_data,
        device MemoryAllocator *mem_alloc,
        const int tid,
        const int grid_size) {
      NodeManager nm;
      nm.nm_data = nm_data;
      nm.mem_alloc = mem_alloc;

      ListManager free_list;
      free_list.lm_data = &(nm.nm_data->free_list);
      free_list.mem_alloc = nm.mem_alloc;

      const int free_size = free_list.num_active();
      const int free_used = atomic_load_explicit(&(nm.nm_data->free_list_used),
                                                 metal::memory_order_relaxed);

      int num_to_copy = 0;
      if (free_used * 2 > free_size) {
        num_to_copy = free_size - free_used;
      } else {
        num_to_copy = free_used;
      }
      const int offs = free_size - num_to_copy;

      using ElemIndex = NodeManager::ElemIndex;
      for (int ii = tid; ii < num_to_copy; ii += grid_size) {
        device auto *dest =
            reinterpret_cast<device ElemIndex *>(free_list.get_ptr(ii));
        *dest = free_list.get<ElemIndex>(ii + offs);
      }
    }

    [[maybe_unused]] void run_gc_reset_free_list(
        device NodeManagerData *nm_data,
        device MemoryAllocator *mem_alloc) {
      NodeManager nm;
      nm.nm_data = nm_data;
      nm.mem_alloc = mem_alloc;

      ListManager free_list;
      free_list.lm_data = &(nm.nm_data->free_list);
      free_list.mem_alloc = nm.mem_alloc;
      const int free_size = free_list.num_active();
      const int free_used = atomic_exchange_explicit(
          &(nm.nm_data->free_list_used), 0, metal::memory_order_relaxed);

      int free_remaining = free_size - free_used;
      free_remaining = free_remaining > 0 ? free_remaining : 0;
      free_list.resize(free_remaining);

      nm.nm_data->recycled_list_size_backup = atomic_exchange_explicit(
          &(nm.nm_data->recycled_list.next), 0, metal::memory_order_relaxed);
    }

    struct GCMoveRecycledToFreeThreadParams {
      int thread_position_in_threadgroup;
      int threadgroup_position_in_grid;
      int threadgroups_per_grid;
      int threads_per_threadgroup;
    };

    [[maybe_unused]] void run_gc_move_recycled_to_free(
        device NodeManagerData *nm_data,
        device MemoryAllocator *mem_alloc,
        thread const GCMoveRecycledToFreeThreadParams &thparams) {
      NodeManager nm;
      nm.nm_data = nm_data;
      nm.mem_alloc = mem_alloc;

      ListManager free_list;
      free_list.lm_data = &(nm.nm_data->free_list);
      free_list.mem_alloc = nm.mem_alloc;

      ListManager recycled_list;
      recycled_list.lm_data = &(nm.nm_data->recycled_list);
      recycled_list.mem_alloc = nm.mem_alloc;

      ListManager data_list;
      data_list.lm_data = &(nm.nm_data->data_list);
      data_list.mem_alloc = nm.mem_alloc;

      const int kInt32Stride = sizeof(int32_t);

      const int recycled_size = nm.nm_data->recycled_list_size_backup;
      using ElemIndex = NodeManager::ElemIndex;
      for (int ii = thparams.threadgroup_position_in_grid; ii < recycled_size;
           ii += thparams.threadgroups_per_grid) {
        const auto elem_idx = recycled_list.get<ElemIndex>(ii);
        device char *ptr = nm.get(elem_idx);
        device const char *ptr_end = ptr + data_list.lm_data->element_stride;
        const int ptr_mod = ((intptr_t)(ptr) % kInt32Stride);
        if (ptr_mod) {
          device char *new_ptr = ptr + kInt32Stride - ptr_mod;
          if (thparams.thread_position_in_threadgroup == 0) {
            for (device char *p = ptr; p < new_ptr; ++p) {
              *p = 0;
            }
          }
          ptr = new_ptr;
        }
        ptr += (thparams.thread_position_in_threadgroup * kInt32Stride);
        while ((ptr + kInt32Stride) <= ptr_end) {
          *reinterpret_cast<device int32_t *>(ptr) = 0;
          ptr += (kInt32Stride * thparams.threads_per_threadgroup);
        }
        while (ptr < ptr_end) {
          *ptr = 0;
          ++ptr;
        }
        if (thparams.thread_position_in_threadgroup == 0) {
          free_list.append(elem_idx);
        }
      }
    })
METAL_END_RUNTIME_UTILS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_UTILS_DEF
#undef METAL_END_RUNTIME_UTILS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
