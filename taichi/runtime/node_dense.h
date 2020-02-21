#pragma once

// Specialized Attributes and functions
struct DenseMeta : public StructMeta {
  bool bitmasked;
  int morton_dim;
};

STRUCT_FIELD(DenseMeta, bitmasked)
STRUCT_FIELD(DenseMeta, morton_dim)

i32 Dense_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void Dense_activate(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto dmeta = (DenseMeta *)meta;
  if (DenseMeta_get_bitmasked(dmeta)) {
    auto element_size = StructMeta_get_element_size(smeta);
    auto num_elements = Dense_get_num_elements(meta, node);
    auto data_section_size = element_size * num_elements;
    auto mask_begin = (uint64 *)(node + data_section_size);
    atomic_or_u64(&mask_begin[i / 64], 1UL << (i % 64));
  }
}

i32 Dense_is_active(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto dmeta = (DenseMeta *)meta;
  if (DenseMeta_get_bitmasked(dmeta)) {
    auto element_size = StructMeta_get_element_size(smeta);
    auto num_elements = Dense_get_num_elements(meta, node);
    auto data_section_size = element_size * num_elements;
    auto mask_begin = node + data_section_size;
    return i32(bool((mask_begin[i / 8] >> (i % 8)) & 1));
  } else {
    return 1;
  }
}

Ptr Dense_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}
