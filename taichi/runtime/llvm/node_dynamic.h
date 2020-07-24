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

void Dynamic_activate(Ptr meta_, Ptr node_, int i) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  // We need to not only update node->n, but also make sure the chunk containing
  // element i is allocated.
  atomic_max_i32(&node->n, i + 1);
  int chunk_start = 0;
  auto p_chunk_ptr = &node->ptr;
  auto chunk_size = meta->chunk_size;
  while (true) {
    if (*p_chunk_ptr == nullptr) {
      locked_task(Ptr(&node->lock), [&] {
        if (*p_chunk_ptr == nullptr) {
          auto rt = meta->context->runtime;
          auto alloc = rt->node_allocators[meta->snode_id];
          *p_chunk_ptr = alloc->allocate();
        }
      });
    }
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
        taichi_printf(rt, "recycling... %p\n", *p_chunk_ptr);
        alloc->recycle(*p_chunk_ptr);
        if (p_chunk_ptr == (Ptr *)*p_chunk_ptr) {
          taichi_printf(rt, "WTH??... %p\n", *p_chunk_ptr);
          return;
        }
        p_chunk_ptr = (Ptr *)*p_chunk_ptr;
      }
      node->ptr = nullptr;
    });
  }
}

i32 Dynamic_append(Ptr meta_, Ptr node_, i32 data) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  auto chunk_size = meta->chunk_size;
  auto i = atomic_add_i32(&node->n, 1);
  int chunk_start = 0;
  auto p_chunk_ptr = &node->ptr;
  while (true) {
    if (*p_chunk_ptr == nullptr) {
      locked_task(Ptr(&node->lock), [&] {
        if (*p_chunk_ptr == nullptr) {
          auto rt = meta->context->runtime;
          auto alloc = rt->node_allocators[meta->snode_id];
          *p_chunk_ptr = alloc->allocate();
        }
      });
    }
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
