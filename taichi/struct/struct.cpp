// Codegen for the hierarchical data structure

#include "struct.h"

namespace taichi {
namespace lang {

void StructCompiler::collect_snodes(SNode &snode) {
  snodes.push_back(&snode);
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    collect_snodes(*ch);
  }
}

}  // namespace lang
}  // namespace taichi
