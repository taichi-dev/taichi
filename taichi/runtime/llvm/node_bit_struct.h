#pragma once

// Specialized Attributes and functions
struct BitStructMeta : public StructMeta {
  int morton_dim;
};

STRUCT_FIELD(BitStructMeta, morton_dim)
// TODO: Correct this file
i32 BitStruct_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void BitStruct_activate(Ptr meta, Ptr node, int i) {
  // Dense elements are always active
}

i32 BitStruct_is_active(Ptr meta, Ptr node, int i) {
  return 1;
}

Ptr BitStruct_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}
