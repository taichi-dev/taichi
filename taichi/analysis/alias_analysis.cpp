#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {

AliasResult alias_analysis(Stmt *var1, Stmt *var2) {
  // If both stmts are allocas, they have the same address iff var1 == var2.
  // If only one of them is an alloca, they can never share the same address.
  if (var1 == var2)
    return AliasResult::same;
  if (!var1 || !var2)
    return AliasResult::different;

  // TODO: further optimize with offset inside PtrOffsetStmt
  // If at least one of var1 and var2 is local, they will be treated here.
  auto retrieve_local = [&](Stmt *var) {
    if (var->is<AllocaStmt>()) {
      return var;
    } else if (var->is<PtrOffsetStmt>() &&
               var->cast<PtrOffsetStmt>()->is_local_ptr()) {
      return var->cast<PtrOffsetStmt>()->origin;
    } else {
      return (Stmt *)nullptr;
    }
  };
  Stmt *origin1 = retrieve_local(var1);
  Stmt *origin2 = retrieve_local(var2);
  if (origin1 != nullptr && origin2 != nullptr) {
    if (origin1 == origin2) {
      if (var1->is<PtrOffsetStmt>() && var2->is<PtrOffsetStmt>()) {
        auto diff = value_diff_ptr_index(var1->cast<PtrOffsetStmt>()->offset,
                                         var2->cast<PtrOffsetStmt>()->offset);
        if (diff.is_diff_certain) {
          return diff.diff_range == 0 ? AliasResult::same
                                      : AliasResult::different;
        }
      }
      return AliasResult::uncertain;
    }
    if (origin1->is<AllocaStmt>() || origin2->is<AllocaStmt>())
      return AliasResult::different;
    TI_ASSERT(origin1->is<GlobalTemporaryStmt>() &&
              origin2->is<GlobalTemporaryStmt>());
    if (origin1->cast<GlobalTemporaryStmt>()->offset ==
        origin2->cast<GlobalTemporaryStmt>()->offset) {
      return AliasResult::uncertain;
    } else {
      return AliasResult::different;
    }
  }
  if (origin1 != nullptr || origin2 != nullptr) {
    return AliasResult::different;
  }

  if (var1->is<AdStackAllocaStmt>() || var2->is<AdStackAllocaStmt>())
    return AliasResult::different;

  // TODO(xumingkuan): Put GlobalTemporaryStmt, ThreadLocalPtrStmt and
  //  BlockLocalPtrStmt into GlobalPtrStmt.
  // If both statements are global temps, they have the same address iff they
  // have the same offset. If only one of them is a global temp, they can never
  // share the same address.
  if (var1->is<GlobalTemporaryStmt>() || var2->is<GlobalTemporaryStmt>()) {
    if (!var1->is<GlobalTemporaryStmt>() || !var2->is<GlobalTemporaryStmt>())
      return AliasResult::different;
    return var1->as<GlobalTemporaryStmt>()->offset ==
                   var2->as<GlobalTemporaryStmt>()->offset
               ? AliasResult::same
               : AliasResult::different;
  }

  if (var1->is<ThreadLocalPtrStmt>() || var2->is<ThreadLocalPtrStmt>()) {
    if (!var1->is<ThreadLocalPtrStmt>() || !var2->is<ThreadLocalPtrStmt>())
      return AliasResult::different;
    return var1->as<ThreadLocalPtrStmt>()->offset ==
                   var2->as<ThreadLocalPtrStmt>()->offset
               ? AliasResult::same
               : AliasResult::different;
  }

  if (var1->is<BlockLocalPtrStmt>() || var2->is<BlockLocalPtrStmt>()) {
    if (!var1->is<BlockLocalPtrStmt>() || !var2->is<BlockLocalPtrStmt>())
      return AliasResult::different;
    return irpass::analysis::same_statements(
               var1->as<BlockLocalPtrStmt>()->offset,
               var2->as<BlockLocalPtrStmt>()->offset)
               ? AliasResult::same
               : AliasResult::uncertain;
  }

  if (var1->is<ExternalPtrStmt>() || var2->is<ExternalPtrStmt>()) {
    if (!var1->is<ExternalPtrStmt>() || !var2->is<ExternalPtrStmt>())
      return AliasResult::different;
    auto ptr1 = var1->as<ExternalPtrStmt>();
    auto ptr2 = var2->as<ExternalPtrStmt>();
    if (ptr1->base_ptr != ptr2->base_ptr) {
      auto base1 = ptr1->base_ptr->as<ArgLoadStmt>();
      auto base2 = ptr2->base_ptr->as<ArgLoadStmt>();
      if (base1->arg_id != base2->arg_id) {
        return AliasResult::different;
      }
    }
    TI_ASSERT(ptr1->indices.size() == ptr2->indices.size());
    bool uncertain = false;
    for (int i = 0; i < (int)ptr1->indices.size(); i++) {
      auto diff = value_diff_ptr_index(ptr1->indices[i], ptr2->indices[i]);
      if (!diff.is_diff_certain) {
        uncertain = true;
      } else if (diff.diff_range != 0) {
        return AliasResult::different;
      }
    }
    return uncertain ? AliasResult::uncertain : AliasResult::same;
  }

  // If both statements are GlobalPtrStmts or GetChStmts, we can check by
  // SNode::id.
  auto get_snode_id = [](Stmt *s) {
    if (auto ptr = s->cast<GlobalPtrStmt>()) {
      return ptr->snode->id;
    } else if (auto get_child = s->cast<GetChStmt>()) {
      return get_child->output_snode->id;
    }
    return -1;
  };
  int snode1 = get_snode_id(var1);
  int snode2 = get_snode_id(var2);
  if (snode1 != -1 && snode2 != -1 && snode1 != snode2) {
    return AliasResult::different;
  }

  // GlobalPtrStmts with guaranteed different indices cannot share the same
  // address.
  if (var1->is<GlobalPtrStmt>() && var2->is<GlobalPtrStmt>()) {
    auto ptr1 = var1->as<GlobalPtrStmt>();
    auto ptr2 = var2->as<GlobalPtrStmt>();
    auto snode = ptr1->snode;
    TI_ASSERT(snode == ptr2->snode);
    TI_ASSERT(ptr1->indices.size() == ptr2->indices.size());
    bool uncertain = false;
    for (int i = 0; i < (int)ptr1->indices.size(); i++) {
      auto diff = value_diff_ptr_index(ptr1->indices[i], ptr2->indices[i]);
      if (!diff.is_diff_certain || (diff.diff_range != 0)) {
        uncertain = true;
      }
      if (std::abs(diff.diff_range) >= 1) {
        return AliasResult::different;
      }
    }
    return uncertain ? AliasResult::uncertain : AliasResult::same;
  }

  // In other cases (probably after lower_access), we don't know if the two
  // statements share the same address.
  return AliasResult::uncertain;
}

bool definitely_same_address(Stmt *var1, Stmt *var2) {
  return alias_analysis(var1, var2) == AliasResult::same;
}

bool maybe_same_address(Stmt *var1, Stmt *var2) {
  return alias_analysis(var1, var2) != AliasResult::different;
}

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
