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
      device ListManager *child_list =
          &(reinterpret_cast<device Runtime *>(runtime_addr)
                ->snode_lists[child_snode_id]);
      clear(child_list);
    }

    kernel void element_listgen(device byte *runtime_addr[[buffer(0)]],
                                device byte *root_addr[[buffer(1)]],
                                device int *args[[buffer(2)]],
                                const uint utid_[[thread_position_in_grid]],
                                const uint grid_size[[threads_per_grid]]) {
      device Runtime *runtime =
          reinterpret_cast<device Runtime *>(runtime_addr);
      device byte *list_data_addr =
          reinterpret_cast<device byte *>(runtime + 1);

      int parent_snode_id = args[0];
      int child_snode_id = args[1];
      device ListManager *parent_list =
          &(runtime->snode_lists[parent_snode_id]);
      device ListManager *child_list = &(runtime->snode_lists[child_snode_id]);
      const SNodeMeta child_meta = runtime->snode_metas[child_snode_id];
      const int child_stride = child_meta.element_stride;
      const int num_slots = child_meta.num_slots;

      for (int ii = utid_; ii < child_list->max_num_elems; ii += grid_size) {
        const int parent_idx = (ii / num_slots);
        if (parent_idx >= num_active(parent_list)) {
          // Since |parent_idx| increases monotonically, we can return directly
          // once it goes beyond the number of active parent elements.
          return;
        }
        const int child_idx = (ii % num_slots);
        const auto parent_elem =
            get<ListgenElement>(parent_list, parent_idx, list_data_addr);
        ListgenElement child_elem;
        child_elem.root_mem_offset = parent_elem.root_mem_offset +
                                     child_idx * child_stride +
                                     child_meta.mem_offset_in_parent;
        if (is_active(root_addr + child_elem.root_mem_offset, child_meta,
                      child_idx)) {
          refine_coordinates(parent_elem,
                             runtime->snode_extractors[child_snode_id],
                             child_idx, &child_elem);
          append(child_list, child_elem, list_data_addr);
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
