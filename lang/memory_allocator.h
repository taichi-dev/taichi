#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"
#include "addr_node.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

struct MemoryAllocator {
  // A tree-like structure that describes the minimal repeating unit in the
  // stream
  Handle<SNode> root;
  bool materialized;

  MemoryAllocator() {
    materialized = false;
    // streams are specialized groups, with discontinuous parts in memory
    root = SNode::create(0);
  }

  SNode &buffer(int id) {
    while ((int)root->ch.size() <= id) {
      root->ch.push_back(SNode::create(1));
      root->ch.back()->buffer_id = root->ch.size() - 1;
    }
    auto ret = root->ch[id];
    return *ret;
  }

  void materialize() {
    if (materialized) {
      return;
    }
    root->materialize();
    root->set();
    materialized = true;
  }

  void print() {
    root->print();
  }
};


}

TC_NAMESPACE_END
