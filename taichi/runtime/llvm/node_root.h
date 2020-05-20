/*******************************************************************************
     Copyright (c) 2020 The Taichi Authors
     Use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

struct RootMeta : public StructMeta {
  int tag;
};

STRUCT_FIELD(RootMeta, tag);

void Root_activate(Ptr meta, Ptr node, int i) {
}

i32 Root_is_active(Ptr meta, Ptr node, int i) {
  return 1;
}

Ptr Root_lookup_element(Ptr meta, Ptr node, int i) {
  // only one element
  return node;
}

i32 Root_get_num_elements(Ptr meta, Ptr node) {
  return 1;
}
