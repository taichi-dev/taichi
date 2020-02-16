#pragma once

// Specialized Attributes and functions
struct pointerMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(pointerMeta, _);

i32 pointer_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void pointer_activate(Ptr meta, Ptr node, int i) {
  auto num_elements = pointer_get_num_elements(meta, node);
  Ptr lock = node + 8*i;
  Ptr &data_ptr = *(Ptr *)(node + 8*(num_elements + i));
  if (data_ptr == nullptr) {
    locked_task(lock, [&] {
      if (data_ptr == nullptr) {
        auto smeta = (StructMeta *)meta;
        auto rt = smeta->context->runtime;
        auto alloc = rt->node_allocators[smeta->snode_id];
        data_ptr = alloc->allocate();
      }
    });
  }
}

void pointer_deactivate(Ptr meta, Ptr node, int i) {
  auto num_elements = pointer_get_num_elements(meta, node);
  Ptr lock = node + 8*i;
  Ptr &data_ptr = *(Ptr *)(node + 8*(num_elements + i));
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

i32 pointer_is_active(Ptr meta, Ptr node, int i) {
  auto num_elements = pointer_get_num_elements(meta, node);
  auto data_ptr = *(Ptr *)(node + 8*(num_elements + i));
  return data_ptr != nullptr;
}

Ptr pointer_lookup_element(Ptr meta, Ptr node, int i) {
  auto num_elements = pointer_get_num_elements(meta, node);
  auto data_ptr = *(Ptr *)(node + 8*(num_elements + i));
  if (data_ptr == nullptr) {
    auto smeta = (StructMeta *)meta;
    auto context = smeta->context;
    data_ptr = (context->runtime)->ambient_elements[smeta->snode_id];
  }
  return data_ptr;
}
