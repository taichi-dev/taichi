#pragma once

// Specialized Attributes and functions
struct BitArrayMeta : public StructMeta {
  int morton_dim;
};

STRUCT_FIELD(BitArrayMeta, morton_dim)

i32 BitArray_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void BitArray_activate(Ptr meta, Ptr node, int i) {
  // Dense elements are always active
}

i32 BitArray_is_active(Ptr meta, Ptr node, int i) {
  return 1;
}

Ptr BitArray_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}
