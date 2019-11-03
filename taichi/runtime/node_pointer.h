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
    data_ptr = (Ptr)(new char[smeta->element_size]);
  }
}

bool Pointer_is_active(Ptr meta, Ptr node, int i) {
  auto data_ptr = *(Ptr *)(node + 0);
  return data_ptr != nullptr;
}

void *Pointer_lookup_element(Ptr meta, Ptr node, int i) {
  return node;
}

int Pointer_get_num_elements(Ptr meta, Ptr node) {
  return 1;
}

