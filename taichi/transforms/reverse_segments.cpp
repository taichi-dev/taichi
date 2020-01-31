#include "../ir.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

namespace irpass {

void reverse_segments(IRNode *root) {
  auto block = dynamic_cast<Block *>(root);
  std::vector<std::vector<pStmt>> statement_blocks(1);
  for (auto &&s : block->statements) {
    if (s->is<FrontendForStmt>()) {
      statement_blocks.emplace_back();
      statement_blocks.back().push_back(std::move(s));
      statement_blocks.emplace_back();
    } else {
      statement_blocks.back().push_back(std::move(s));
    }
  }
  block->statements.clear();
  std::reverse(statement_blocks.begin(), statement_blocks.end());
  for (auto &sblock : statement_blocks) {
    for (auto &&s : sblock) {
      block->statements.push_back(std::move(s));
    }
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
