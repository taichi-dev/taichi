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
  if (var1->is<AllocaStmt>() || var2->is<AllocaStmt>())
    return AliasResult::different;
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

  // If both statements are GlobalPtrStmts or GetChStmts, we can check by
  // SNode::id.
  TI_ASSERT(var1->width() == 1);
  TI_ASSERT(var2->width() == 1);
  auto get_snode_id = [](Stmt *s) {
    if (auto ptr = s->cast<GlobalPtrStmt>()) {
      return ptr->snodes[0]->id;
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
    auto snode = ptr1->snodes[0];
    TI_ASSERT(snode == ptr2->snodes[0]);
    TI_ASSERT(ptr1->indices.size() == ptr2->indices.size());
    bool uncertain = false;
    for (int i = 0; i < (int)ptr1->indices.size(); i++) {
      auto diff = value_diff_ptr_index(ptr1->indices[i], ptr2->indices[i]);
      if (!diff.is_diff_certain || (diff.diff_range != 0)) {
        uncertain = true;
      }
      if (std::abs(diff.diff_range) >=
          (1 << snode->extractors[snode->physical_index_position[i]]
                    .trailing_bits)) {
        // For `trailing_bits == 2`, if the difference of the two indices >= 4,
        // we are sure that they point to the different address.
        //
        // However, if two indices are different by < 4, we are uncertain in
        // this case. E.g., `idx1 = 0` and `idx2 = 2` point to the same
        // cell, but `idx1 = 3` and `idx2 = 5` point to the different cells.
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
