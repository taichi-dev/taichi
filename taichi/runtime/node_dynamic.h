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
  locked_task(Ptr(&node->lock), [&] {
    if (i < node->n)
      return;
    node->n = i + 1;
    int chunk_start = 0;
    auto p_chunk_ptr = &node->ptr;
    auto chunk_size = meta->chunk_size;
    auto rt = (Runtime *)meta->context->runtime;
    auto alloc = rt->node_allocators[meta->snode_id];
    while (true) {
      if (*p_chunk_ptr == nullptr) {
        *p_chunk_ptr = NodeAllocator_allocate(alloc);
      }
      if (i < chunk_start + chunk_size) {
        return;
      }
      p_chunk_ptr = (Ptr *)*p_chunk_ptr;
      chunk_start += chunk_size;
    }
  });
}

void Dynamic_append(Ptr meta_, Ptr node_, i32 data) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  auto chunk_size = meta->chunk_size;
  locked_task(Ptr(&node->lock), [&] {
    auto i = node->n;
    int chunk_start = 0;
    auto p_chunk_ptr = &node->ptr;
    auto rt = (Runtime *)meta->context->runtime;
    auto alloc = rt->node_allocators[meta->snode_id];
    while (true) {
      if (*p_chunk_ptr == nullptr) {
        *p_chunk_ptr = NodeAllocator_allocate(alloc);
      }
      if (i < chunk_start + chunk_size) {
        node->n += 1;
        *(i32 *)(*p_chunk_ptr + sizeof(Ptr) +
                 (i - chunk_start) * meta->element_size) = data;
        return;
      }
      p_chunk_ptr = (Ptr *)*p_chunk_ptr;
      chunk_start += chunk_size;
    }
  });
}

bool Dynamic_is_active(Ptr meta_, Ptr node_, int i) {
  auto node = (DynamicNode *)(node_);
  return i < node->n;
}

void *Dynamic_lookup_element(Ptr meta_, Ptr node_, int i) {
  auto meta = (DynamicMeta *)(meta_);
  auto node = (DynamicNode *)(node_);
  int chunk_start = 0;
  auto chunk_ptr = node->ptr;
  auto chunk_size = meta->chunk_size;
  while (true) {
    if (i < chunk_start + chunk_size) {
      return chunk_ptr + sizeof(Ptr) + (i - chunk_start) * meta->element_size;
    }
    chunk_ptr = *(Ptr *)chunk_ptr;
    chunk_start += chunk_size;
  }
  return nullptr;
}

int Dynamic_get_num_elements(Ptr meta_, Ptr node_) {
  auto node = (DynamicNode *)(node_);
  return node->n;
}
