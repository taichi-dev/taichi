#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {

// Returns the set of SNodes that are read or written
std::pair<std::unordered_set<SNode *>, std::unordered_set<SNode *>>
gather_snode_read_writes(IRNode *root) {
  std::pair<std::unordered_set<SNode *>, std::unordered_set<SNode *>> accessed;
  irpass::analysis::gather_statements(root, [&](Stmt *stmt) {
    Stmt *ptr = nullptr;
    bool read = false, write = false;
    if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      read = true;
      ptr = global_load->src;
    } else if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      write = true;
      ptr = global_store->dest;
    } else if (auto global_atomic = stmt->cast<AtomicOpStmt>()) {
      read = true;
      write = true;
      ptr = global_atomic->dest;
    }
    if (ptr) {
      if (auto *global_ptr = ptr->cast<GlobalPtrStmt>()) {
        if (read)
          accessed.first.emplace(global_ptr->snode);
        if (write)
          accessed.second.emplace(global_ptr->snode);
      }
    }
    return false;
  });
  return accessed;
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
