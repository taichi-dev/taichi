#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

struct MemoryAllocator {
  // A tree-like structure that describes the minimal repeating unit in the
  // stream
  Handle<AddrNode> root;

  MemoryAllocator() {
    // streams are specialized groups, with discontinuous parts in memory
    root = AddrNode::create(0);
  }

  AddrNode &buffer(int id) {
    while ((int)root->ch.size() <= id) {
      root->ch.push_back(AddrNode::create(1));
      root->ch.back()->buffer_id = root->ch.size() - 1;
    }
    auto ret = root->ch[id];
    return *ret;
  }

  void materialize() {
    root->materialize();
    root->set();
  }

  void print() {
    root->print();
  }
};


}

TC_NAMESPACE_END
