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

// Just a mock to illustrate what the Runtime looks like, do not use.
// The actual Runtime struct has to be emitted by codegen, because it depends
// on the number of SNodes.
struct Runtime {
  SNodeMeta *snode_metas;
  SNodeExtractors *snode_extractors;
  ListManager *snode_lists;
};

#define METAL_BEGIN_RUNTIME_KERNELS_DEF
#define METAL_END_RUNTIME_KERNELS_DEF

#endif  // TI_INSIDE_METAL_CODEGEN

// clang-format off
METAL_BEGIN_RUNTIME_KERNELS_DEF
STR(
    // clang-format on
    kernel void clear_list(device byte *runtime_addr[[buffer(0)]],
                           device int *args[[buffer(1)]],
                           const uint utid_[[thread_position_in_grid]]) {
      if (utid_ > 0)
        return;
      int child_snode_id = args[1];
      ListManager child_list;
      child_list.lm_data =
          (reinterpret_cast<device Runtime *>(runtime_addr)->snode_lists +
           child_snode_id);
      child_list.clear();
    }

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
      // |max_num_elems| is NOT padded to power-of-two, while |num_slots| is.
      // So we need to cap the loop precisely at child's |max_num_elems|.
      const int max_num_elems = args[2];
      for (int ii = utid_; ii < max_num_elems; ii += grid_size) {
        const int parent_idx = (ii / num_slots);
        if (parent_idx >= parent_list.num_active()) {
          // Since |parent_idx| increases monotonically, we can return directly
          // once it goes beyond the number of active parent elements.
          return;
        }
        const int child_idx = (ii % num_slots);
        const auto parent_elem = parent_list.get<ListgenElement>(parent_idx);
        device byte *parent_addr = root_addr + parent_elem.root_mem_offset;
        if (is_active(parent_addr, parent_meta, child_idx)) {
          ListgenElement child_elem;
          child_elem.root_mem_offset = parent_elem.root_mem_offset +
                                       child_idx * child_stride +
                                       child_meta.mem_offset_in_parent;
          refine_coordinates(parent_elem,
                             runtime->snode_extractors[parent_snode_id],
                             child_idx, &child_elem);
          child_list.append(child_elem);
        }
      }
    }
    // clang-format off
)
METAL_END_RUNTIME_KERNELS_DEF
// clang-format on

#undef METAL_BEGIN_RUNTIME_KERNELS_DEF
#undef METAL_END_RUNTIME_KERNELS_DEF

#include "taichi/backends/metal/shaders/epilog.h"
