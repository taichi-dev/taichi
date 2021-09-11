#ifndef TI_METAL_NESTED_INCLUDE

#define TI_METAL_NESTED_INCLUDE
#include "taichi/backends/metal/shaders/runtime_utils.metal.h"
#undef TI_METAL_NESTED_INCLUDE

#else
#include "taichi/backends/metal/shaders/runtime_utils.metal.h"
#endif  // TI_METAL_NESTED_INCLUDE

#include "taichi/backends/metal/shaders/prolog.h"

#ifdef TI_INSIDE_METAL_CODEGEN

#ifndef TI_METAL_NESTED_INCLUDE
#define METAL_BEGIN_RUNTIME_KERNELS_DEF \
  constexpr auto kMetalRuntimeKernelsSourceCode =
#define METAL_END_RUNTIME_KERNELS_DEF ;
#else
#define METAL_BEGIN_RUNTIME_KERNELS_DEF
#define METAL_END_RUNTIME_KERNELS_DEF
#endif  // TI_METAL_NESTED_INCLUDE

#else

static_assert(false, "Do not include");

#define METAL_BEGIN_RUNTIME_KERNELS_DEF
#define METAL_END_RUNTIME_KERNELS_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

// clang-format off
METAL_BEGIN_RUNTIME_KERNELS_DEF
STR(
    kernel void element_listgen(device byte *runtime_addr[[buffer(0)]],
                                device byte *root_addr[[buffer(1)]],
                                device int *args[[buffer(2)]],
                                const uint utid_[[thread_position_in_grid]],
                                const uint grid_size[[threads_per_grid]]) {
      device Runtime *runtime =
          reinterpret_cast<device Runtime *>(runtime_addr);
      device MemoryAllocator *mem_alloc =
          reinterpret_cast<device MemoryAllocator *>(runtime + 1);

      const int parent_snode_id = args[0];
      const int child_snode_id = args[1];
      ListManager parent_list;
      parent_list.lm_data = (runtime->snode_lists + parent_snode_id);
      parent_list.mem_alloc = mem_alloc;
      ListManager child_list;
      child_list.lm_data = (runtime->snode_lists + child_snode_id);
      child_list.mem_alloc = mem_alloc;
      const SNodeMeta parent_meta = runtime->snode_metas[parent_snode_id];
      const int child_stride = parent_meta.element_stride;
      const int num_slots = parent_meta.num_slots;
      const SNodeMeta child_meta = runtime->snode_metas[child_snode_id];
      // No need to put an upperbound here (same for listgen and struct-for).
      // Say the number of valid elements per child container is 5, which then
      // gets padded to 8 (POT). is_active() knows how to properly skip those.
      for (int ii = utid_;; ii += grid_size) {
        const int parent_idx = (ii / num_slots);
        if (parent_idx >= parent_list.num_active()) {
          // Since |parent_idx| increases monotonically, we can return directly
          // once it goes beyond the number of active parent elements.
          return;
        }
        const int child_idx = (ii % num_slots);
        const auto parent_elem = parent_list.get<ListgenElement>(parent_idx);
        device byte *parent_addr =
            mtl_lgen_snode_addr(parent_elem, root_addr, runtime, mem_alloc);
        if (is_active(parent_addr, parent_meta, child_idx)) {
          ListgenElement child_elem;
          if (parent_meta.type != SNodeMeta::Pointer) {
            // Need to inherit |parent_elem|'s NodeManager settings. This means
            // that the memory of the child cell will be part of that of the
            // parent container.
            //
            // For example, denoting parent as Y, and child as Z:
            //
            // * Both Y and Z are `dense`: belonged_nodemgr.id = -1. This is the
            //   simplest case, both Y and Z's memory location are known at
            //   compile time. ==> Both live in the `root` buffer.
            // * Y's parent (X) is a `pointer`: Y's NodeManager ID >= 0 (i.e.
            //   |parent_elem.belonged_nodemgr| >= 0), Z inherits Y's
            //   NodeManager settings. ==> Both Y and Z are in the memory
            //   dynamically allocated for X.
            // * Y is `dense`, but Z is a `pointer`: Z's memory location is
            //   known at compile time! So Z itself still lives in the `root`
            //   buffer! However, each Z cell stores the pointer allocated from
            //   the runtime memory pool.
            child_elem.belonged_nodemgr = parent_elem.belonged_nodemgr;
            child_elem.mem_offset =
                parent_elem.mem_offset + child_idx * child_stride;
          } else {
            child_elem.belonged_nodemgr.id = parent_snode_id;
            child_elem.belonged_nodemgr.elem_idx =
                SNodeRep_pointer::to_nodemgr_idx(parent_addr, child_idx);
            // For `pointer` parent, |child_elem.belonged_nodemgr.elem_idx|
            // has already encoded the memory offset for which this child cell
            // belongs to. Therefore |mem_offset| is just 0.
            child_elem.mem_offset = 0;
          }
          child_elem.mem_offset += child_meta.mem_offset_in_parent;

          refine_coordinates(parent_elem.coords,
                             runtime->snode_extractors[parent_snode_id],
                             child_idx, &(child_elem.coords));
          child_list.append(child_elem);
        }
      }
    }

    kernel void gc_compact_free_list(
        device byte *runtime_addr [[buffer(0)]], device int *args [[buffer(1)]],
        const uint utid_ [[thread_position_in_grid]],
        const uint grid_size [[threads_per_grid]]) {
      device Runtime *runtime =
          reinterpret_cast<device Runtime *>(runtime_addr);
      device MemoryAllocator *mem_alloc =
          reinterpret_cast<device MemoryAllocator *>(runtime + 1);
      const int snode_id = args[0];
      run_gc_compact_free_list((runtime->snode_allocators + snode_id),
                               mem_alloc, utid_, grid_size);
    }

    kernel void gc_reset_free_list(device byte *runtime_addr [[buffer(0)]],
                                   device int *args [[buffer(1)]],
                                   const uint utid_
                                   [[thread_position_in_grid]]) {
      if (utid_ > 0) return;

      device Runtime *runtime =
          reinterpret_cast<device Runtime *>(runtime_addr);
      device MemoryAllocator *mem_alloc =
          reinterpret_cast<device MemoryAllocator *>(runtime + 1);
      const int snode_id = args[0];
      run_gc_reset_free_list((runtime->snode_allocators + snode_id), mem_alloc);
    }

    kernel void gc_move_recycled_to_free(
        device byte *runtime_addr [[buffer(0)]], device int *args [[buffer(1)]],
        const uint utid_in_tg_ [[thread_position_in_threadgroup]],
        const uint utgid_ [[threadgroup_position_in_grid]],
        const uint tg_per_grid [[threadgroups_per_grid]],
        const uint threads_per_tg [[threads_per_threadgroup]]) {
      device Runtime *runtime =
          reinterpret_cast<device Runtime *>(runtime_addr);
      device MemoryAllocator *mem_alloc =
          reinterpret_cast<device MemoryAllocator *>(runtime + 1);
      const int snode_id = args[0];

      GCMoveRecycledToFreeThreadParams thparams;
      thparams.thread_position_in_threadgroup = utid_in_tg_;
      thparams.threadgroup_position_in_grid = utgid_;
      thparams.threadgroups_per_grid = tg_per_grid;
      thparams.threads_per_threadgroup = threads_per_tg;
      run_gc_move_recycled_to_free((runtime->snode_allocators + snode_id),
                                   mem_alloc, thparams);
    })
METAL_END_RUNTIME_KERNELS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_KERNELS_DEF
#undef METAL_END_RUNTIME_KERNELS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
