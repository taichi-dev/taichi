#include "../ir.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

namespace irpass {

void reverse_offloads(IRNode *root) {
  auto block = dynamic_cast<Block *>(root);
  for (auto &s : block->statements) {
    TC_ASSERT(s->is<OffloadedStmt>());
  }
  std::reverse(block->statements.begin(), block->statements.end());
}

}

TLANG_NAMESPACE_END
