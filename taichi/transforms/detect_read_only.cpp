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
      auto rec = std::make_pair(SNodeAccessFlag::read_only, snode);
      if (std::find(offload->scratch_opt.begin(), offload->scratch_opt.end(),
                    rec) == offload->scratch_opt.end()) {
        offload->scratch_opt.push_back(rec);
      }
    }
  }
}

}  // namespace

void detect_read_only(IRNode *root) {
  for (auto &offload : root->as<Block>()->statements) {
    detect_read_only_in_task(offload->as<OffloadedStmt>());
  }
}

}  // namespace irpass

TLANG_NAMESPACE_END
