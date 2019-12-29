#include "../ir.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

namespace irpass {

void reverse_offloads(IRNode *root) {
  // Note: clear_list and listgen statements should not be mechanically reversed.
  // Their movement should follow the next non-auxiliary offloaded statement.
  auto block = dynamic_cast<Block *>(root);
  std::vector<std::vector<pStmt>> statement_blocks(1);
  for (auto &s : block->statements) {
    TC_ASSERT(s->is<OffloadedStmt>());
  }
  for (auto &&s_ : block->statements) {
    auto s = s_->as<OffloadedStmt>();
    bool is_aux = s->task_type == OffloadedStmt::TaskType::clear_list ||
                  s->task_type == OffloadedStmt::TaskType::listgen;
    statement_blocks.back().push_back(std::move(s_));
    if (!is_aux) {
      statement_blocks.emplace_back();
    }
  }
  block->statements.clear();
  std::reverse(statement_blocks.begin(), statement_blocks.end());
  for (auto &sblock: statement_blocks) {
    for (auto &&s: sblock) {
      block->statements.push_back(std::move(s));
    }
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
