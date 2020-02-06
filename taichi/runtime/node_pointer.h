#pragma once

// Specialized Attributes and functions
struct PointerMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(PointerMeta, _);

void Pointer_activate(Ptr meta, Ptr node, int i) {
  Ptr lock = node;
  Ptr &data_ptr = *(Ptr *)(node + 8);
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

void Pointer_deactivate(Ptr meta, Ptr node) {
  Ptr lock = node;
  Ptr &data_ptr = *(Ptr *)(node + 8);
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

bool Pointer_is_active(Ptr meta, Ptr node, int i) {
  auto data_ptr = *(Ptr *)(node + 8);
  return data_ptr != nullptr;
}

void *Pointer_lookup_element(Ptr meta, Ptr node, int i) {
  auto data_ptr = *(Ptr *)(node + 8);
  if (data_ptr == nullptr) {
    auto smeta = (StructMeta *)meta;
    auto context = smeta->context;
    data_ptr = (context->runtime)->ambient_elements[smeta->snode_id];
  }
  return data_ptr;
}

int Pointer_get_num_elements(Ptr meta, Ptr node) {
  return 1;
}
