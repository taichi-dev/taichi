#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/transforms.h"

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

class ExternalPtrAccessVisitor : public BasicStmtVisitor {
 private:
  std::unordered_map<int, ExternalPtrAccess> &map_;

 public:
  using BasicStmtVisitor::visit;

  ExternalPtrAccessVisitor(std::unordered_map<int, ExternalPtrAccess> &map)
      : BasicStmtVisitor(), map_(map) {
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (!(stmt->src && stmt->src->is<ExternalPtrStmt>()))
      return;

    ExternalPtrStmt *src = stmt->src->cast<ExternalPtrStmt>();
    ArgLoadStmt *arg = src->base_ptr->cast<ArgLoadStmt>();
    if (map_.find(arg->arg_id) != map_.end()) {
      map_[arg->arg_id] = map_[arg->arg_id] | ExternalPtrAccess::READ;
    } else {
      map_[arg->arg_id] = ExternalPtrAccess::READ;
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (!(stmt->dest && stmt->dest->is<ExternalPtrStmt>()))
      return;

    ExternalPtrStmt *dst = stmt->dest->cast<ExternalPtrStmt>();
    ArgLoadStmt *arg = dst->base_ptr->cast<ArgLoadStmt>();
    if (map_.find(arg->arg_id) != map_.end()) {
      map_[arg->arg_id] = map_[arg->arg_id] | ExternalPtrAccess::WRITE;
    } else {
      map_[arg->arg_id] = ExternalPtrAccess::WRITE;
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!(stmt->dest && stmt->dest->is<ExternalPtrStmt>()))
      return;

    // Atomics modifies existing state (therefore both read & write)
    ExternalPtrStmt *dst = stmt->dest->cast<ExternalPtrStmt>();
    ArgLoadStmt *arg = dst->base_ptr->cast<ArgLoadStmt>();
    map_[arg->arg_id] = ExternalPtrAccess::WRITE | ExternalPtrAccess::READ;
  }
};

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

std::unordered_map<int, ExternalPtrAccess> detect_external_ptr_access_in_task(
    OffloadedStmt *offload) {
  std::unordered_map<int, ExternalPtrAccess> map;
  ExternalPtrAccessVisitor v(map);
  offload->accept(&v);
  return map;
}

}  // namespace irpass

TLANG_NAMESPACE_END
