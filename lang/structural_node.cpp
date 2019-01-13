#include "structural_node.h"
#include "math.h"

TLANG_NAMESPACE_BEGIN

int SNode::counter = 0;

SNode &SNode::place(Matrix &mat) {
  for (auto &e: mat.entries) {
    this->place(e);
  }
}

TLANG_NAMESPACE_END