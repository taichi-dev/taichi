#pragma once

// Specialized Attributes and functions
struct PointerMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(PointerMeta, _);

i32 Pointer_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

bool is_representative(uint32 mask, uint64 value) {
#if defined(ARCH_cuda)
  // If many threads in the mask share the same value, simply
  // elect one thread to return true and let others return false.
  if (cuda_compute_capability() < 70) {
    // <= Pascal
    bool has_following_eqiv = false;
    for (int s = 1; s < 32; s++) {
      auto cond = warp_idx() + s < 32 && ((mask >> (warp_idx() + s)) & 1);
#define TEST_PEER(x) ((x) == cuda_shfl_down_sync_i32(mask, (x), s, 31))
      auto equiv = cond && TEST_PEER(i32(i64(value))) &&
                   TEST_PEER(i32((u64)value >> 32));
#undef TEST_PEER
      has_following_eqiv = has_following_eqiv || equiv;
    }
    return !has_following_eqiv;
  } else {
    // >= Volta > Pascal
    i32 equiv_mask = cuda_match_any_sync_i64(mask, i64(value));
    auto leader = cttz_i32(equiv_mask);
    return warp_idx() == leader;
  }
#else
  return true;
#endif
}

void Pointer_activate(Ptr meta_, Ptr node, int i) {
  auto meta = (StructMeta *)meta_;
  auto num_elements = Pointer_get_num_elements(meta_, node);
  volatile Ptr lock = node + 8 * i;
  volatile Ptr *data_ptr = (Ptr *)(node + 8 * (num_elements + i));

  if (*data_ptr == nullptr) {
    // The cuda_ calls will return 0 or do noop on CPUs
    u32 mask = cuda_active_mask();
    if (is_representative(mask, (u64)lock)) {
      locked_task(lock,
                  [&] {
                    auto rt = meta->context->runtime;
                    auto alloc = rt->node_allocators[meta->snode_id];
                    auto allocated = (u64)alloc->allocate();
                    // TODO: Not sure if we really need atomic_exchange here,
                    // just to be safe.
                    atomic_exchange_u64((u64 *)data_ptr, allocated);
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
  auto smeta = (StructMeta *)meta;
  auto rt = smeta->context->runtime;
  if (data_ptr != nullptr) {
    locked_task(lock, [&] {
      if (data_ptr != nullptr) {
        auto alloc = rt->node_allocators[smeta->snode_id];
        alloc->recycle(data_ptr);
        /*
        if (smeta->snode_id == 1) {
          taichi_printf(
              rt,
              "deallocating snode %d... %d node %p lock %p num_elements %d\n",
              smeta->snode_id, i, node, lock, num_elements);
        }
        */
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
