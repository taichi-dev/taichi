#pragma once

// Specialized Attributes and functions
struct PointerMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(PointerMeta, _);

i32 Pointer_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void Pointer_activate(Ptr meta_, Ptr node, int i) {
  auto meta = (StructMeta *)meta_;
  auto num_elements = Pointer_get_num_elements(meta_, node);
  Ptr lock = node + 8 * i;
  Ptr volatile *data_ptr = (Ptr *)(node + 8 * (num_elements + i));

  if (*data_ptr == nullptr) {
    i32 mask = cuda_active_mask();
#if CUDA_CC < 70
    bool has_following_eqiv = false;
    for (int s = 1; s < 32; s++) {
#define TEST(x) ((x) == cuda_shfl_down_sync_i32(mask, (x), s, 31))
      auto cond = warp_idx() + s < 32 && ((mask >> (warp_idx() + s)) & 1);
      auto equiv = cond && TEST(i32(i64(lock))) && TEST(i32((u64)lock >> 32));
      has_following_eqiv = has_following_eqiv || equiv;
#undef TEST
    }
    bool needs_activation = !has_following_eqiv;
#else
    // Volta +
    i32 equiv_mask = cuda_match_any_sync_i64(mask, i64(lock));
    auto leader = cttz_i32(equiv_mask);
    bool needs_activation = warp_idx() == leader;
#endif
    if (needs_activation) {
      locked_task(lock,
                  [&] {
                    auto rt = meta->context->runtime;
                    auto alloc = rt->node_allocators[meta->snode_id];
                    *data_ptr = alloc->allocate();
                  },
                  [&]() { return *data_ptr == nullptr; });
    }
    warp_barrier(mask);
  }
}

void Pointer_deactivate(Ptr meta, Ptr node, int i) {
  auto num_elements = Pointer_get_num_elements(meta, node);
  Ptr lock = node + 8 * i;
  Ptr &data_ptr = *(Ptr *)(node + 8 * (num_elements + i));
  if (data_ptr != nullptr) {
    locked_task(lock, [&] {
      if (data_ptr != nullptr) {
        auto smeta = (StructMeta *)meta;
        auto rt = smeta->context->runtime;
        auto alloc = rt->node_allocators[smeta->snode_id];
        alloc->recycle(data_ptr);
        data_ptr = nullptr;
      }
    });
  }
}

i32 Pointer_is_active(Ptr meta, Ptr node, int i) {
  auto num_elements = Pointer_get_num_elements(meta, node);
  auto data_ptr = *(Ptr *)(node + 8 * (num_elements + i));
  return data_ptr != nullptr;
}

Ptr Pointer_lookup_element(Ptr meta, Ptr node, int i) {
  auto num_elements = Pointer_get_num_elements(meta, node);
  auto data_ptr = *(Ptr *)(node + 8 * (num_elements + i));
  if (data_ptr == nullptr) {
    auto smeta = (StructMeta *)meta;
    auto context = smeta->context;
    data_ptr = (context->runtime)->ambient_elements[smeta->snode_id];
  }
  return data_ptr;
}
