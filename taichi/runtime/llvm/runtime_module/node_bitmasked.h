#pragma once

// Specialized Attributes and functions
struct BitmaskedMeta : public StructMeta {
  bool _;
};

STRUCT_FIELD(BitmaskedMeta, _);

i32 Bitmasked_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void Bitmasked_activate(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Bitmasked_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = (u32 *)(node + data_section_size);
  atomic_or_u32(&mask_begin[i / 32], 1UL << (i % 32));
}

void Bitmasked_deactivate(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Bitmasked_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = (u32 *)(node + data_section_size);
  atomic_and_u32(&mask_begin[i / 32], ~(1UL << (i % 32)));
}

i32 Bitmasked_is_active(Ptr meta, Ptr node, int i) {
  auto smeta = (StructMeta *)meta;
  auto element_size = StructMeta_get_element_size(smeta);
  auto num_elements = Dense_get_num_elements(meta, node);
  auto data_section_size = element_size * num_elements;
  auto mask_begin = node + data_section_size;
  return i32(bool((mask_begin[i / 8] >> (i % 8)) & 1));
}

Ptr Bitmasked_lookup_element(Ptr meta, Ptr node, int i) {
  return node + ((StructMeta *)meta)->element_size * i;
}
