#include "../ir.h"

TLANG_NAMESPACE_BEGIN

VecStatement convert_to_range_for(StructForStmt *struct_for) {
  // auto range_for = Stmt::make<RangeForStmt>();
}

namespace irpass {
void demote_dense_struct_fors(IRNode *root) {
  auto *block = dynamic_cast<Block *>(root);
  std::vector<Stmt *> block_body;
  for (int i = 0; i < (int)block->statements.size(); i++) {
    block_body.push_back(block->statements[i].get());
  }
  for (int i = 0; i < (int)block_body.size(); i++) {
    auto s_ = block_body[i];
    if (auto s = s_->as<StructForStmt>()) {
      auto snode = s->snode;
      bool all_dense = true;
      while (snode) {
        if (snode->type != SNodeType::dense) {
          all_dense = false;
        }
        snode = snode->parent;
      }
      if (all_dense) {
        s->parent->replace_with(s, convert_to_range_for(s), false);
      }
    }
  }
}
}  // namespace irpass

TLANG_NAMESPACE_END
