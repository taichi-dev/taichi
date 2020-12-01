#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"

#include <algorithm>

TLANG_NAMESPACE_BEGIN

namespace irpass {

namespace {

void detect_read_only_in_task(OffloadedStmt *offload) {
  auto accessed = irpass::analysis::gather_snode_read_writes(offload);
  for (auto snode : accessed.first) {
    if (accessed.second.count(snode) == 0) {
      // read-only SNode
      offload->mem_access_opt.add_flag(snode, SNodeAccessFlag::read_only);
    }
  }
}

}  // namespace

void detect_read_only(IRNode *root) {
  if (root->is<Block>()) {
    for (auto &offload : root->as<Block>()->statements) {
      detect_read_only_in_task(offload->as<OffloadedStmt>());
    }
  } else {
    detect_read_only_in_task(root->as<OffloadedStmt>());
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
