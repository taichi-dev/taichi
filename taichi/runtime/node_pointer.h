#pragma once

// Specialized Attributes and functions
struct PointerMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(PointerMeta, _);

void Pointer_activate(Ptr meta, Ptr node, int i) {
  Ptr &data_ptr = *(Ptr *)(node + 0);
  if (data_ptr == nullptr) {
    auto smeta = (StructMeta *)meta;
    auto rt = (Runtime *)smeta->context->runtime;
    auto alloc = rt->node_allocators[smeta->snode_id];
    data_ptr = NodeAllocator_allocate(alloc);
  }
}

bool Pointer_is_active(Ptr meta, Ptr node, int i) {
  auto data_ptr = *(Ptr *)(node + 0);
  return data_ptr != nullptr;
}

void *Pointer_lookup_element(Ptr meta, Ptr node, int i) {
  auto data_ptr = *(Ptr *)(node + 0);
  if (data_ptr == nullptr) {
    auto smeta = (StructMeta *)meta;
    auto context = smeta->context;
    data_ptr = ((Runtime *)context->runtime)->ambient_elements[smeta->snode_id];
  }
  return data_ptr;
}

int Pointer_get_num_elements(Ptr meta, Ptr node) { return 1; }
