#pragma once

struct DynamicNode {
  i32 lock;
  i32 n;
  Ptr ptr;
};

// Specialized Attributes and functions
struct DynamicMeta : public StructMeta {
  int chunk_size;
};

STRUCT_FIELD(DynamicMeta, chunk_size);

void Dynamic_get_or_allocate_chunk_once(DynamicMeta *meta,
                                        DynamicNode *node,
                                        Ptr *p_chunk_ptr) {
  // Double checked locking pattern requires memfence being inserted, see
  // https://www.aristeia.com/Papers/DDJ_Jul_Aug_2004_revised.pdf
  // https://stackoverflow.com/questions/33115687/acquire-barrier-in-the-double-checked-locking-pattern
  //
  // Ideally, we should implement this pattern following the one shown in
  // https://en.wikipedia.org/wiki/Double-checked_locking#Usage_in_C++11, i.e.,
  // using store_release and load_acquire to form the sync-with relationship.
  // However, these instructions are only available in PTX at this point.
  // https://stackoverflow.com/a/54924933/12003165
  Ptr tmp_ptr = *p_chunk_ptr;
  grid_memfence();
  if (tmp_ptr == nullptr) {
    locked_task(Ptr(&node->lock), [meta, p_chunk_ptr, &tmp_ptr] {
      tmp_ptr = *p_chunk_ptr;
      if (tmp_ptr == nullptr) {
        auto rt = meta->context->runtime;
        auto alloc = rt->node_allocators[meta->snode_id];
        tmp_ptr = alloc->allocate();
        grid_memfence();
        *p_chunk_ptr = tmp_ptr;
      }
    });
  }
}

void Dynamic_activate(Ptr meta_, Ptr node_, int i) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  // We need to not only update node->n, but also make sure the chunk containing
  // element i is allocated.
  atomic_max_i32(&node->n, i + 1);
  int chunk_start = 0;
  Ptr *p_chunk_ptr = &node->ptr;
  const auto chunk_size = meta->chunk_size;
  while (true) {
    Dynamic_get_or_allocate_chunk_once(meta, node, p_chunk_ptr);
    if (i < chunk_start + chunk_size) {
      return;
    }
    p_chunk_ptr = (Ptr *)*p_chunk_ptr;
    chunk_start += chunk_size;
  }
}

void Dynamic_deactivate(Ptr meta_, Ptr node_) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  if (node->n > 0) {
    locked_task(Ptr(&node->lock), [&] {
      node->n = 0;
      auto p_chunk_ptr = &node->ptr;
      auto rt = meta->context->runtime;
      auto alloc = rt->node_allocators[meta->snode_id];
      while (*p_chunk_ptr) {
        alloc->recycle(*p_chunk_ptr);
        p_chunk_ptr = (Ptr *)*p_chunk_ptr;
      }
      node->ptr = nullptr;
    });
  }
}

i32 Dynamic_append(Ptr meta_, Ptr node_, i32 data) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  const auto chunk_size = meta->chunk_size;
  auto i = atomic_add_i32(&node->n, 1);
  int chunk_start = 0;
  Ptr *p_chunk_ptr = &node->ptr;
  while (true) {
    Dynamic_get_or_allocate_chunk_once(meta, node, p_chunk_ptr);
    if (i < chunk_start + chunk_size) {
      *(i32 *)(*p_chunk_ptr + sizeof(Ptr) +
               (i - chunk_start) * meta->element_size) = data;
      break;
    }
    p_chunk_ptr = (Ptr *)(*p_chunk_ptr);
    chunk_start += chunk_size;
  }
  return i;
}

i32 Dynamic_is_active(Ptr meta_, Ptr node_, int i) {
  auto node = (DynamicNode *)(node_);
  return i32(i < node->n);
}

Ptr Dynamic_lookup_element(Ptr meta_, Ptr node_, int i) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  if (Dynamic_is_active(meta_, node_, i)) {
    int chunk_start = 0;
    auto chunk_ptr = node->ptr;
    auto chunk_size = meta->chunk_size;
    while (true) {
      if (i < chunk_start + chunk_size) {
        auto addr =
            chunk_ptr + sizeof(Ptr) + (i - chunk_start) * meta->element_size;
        return addr;
      }
      chunk_ptr = *(Ptr *)chunk_ptr;
      chunk_start += chunk_size;
    }
  } else {
    return (meta->context->runtime)->ambient_elements[meta->snode_id];
  }
}

i32 Dynamic_get_num_elements(Ptr meta_, Ptr node_) {
  auto node = (DynamicNode *)(node_);
  return node->n;
}
