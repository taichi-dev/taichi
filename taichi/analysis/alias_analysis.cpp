#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

AliasResult same_value(Stmt *val1, Stmt *val2) {
  if (val1 == val2)
    return AliasResult::same;
  if (!val1 || !val2)
    return AliasResult::different;
  if (val1->is<ConstStmt>() && val2->is<ConstStmt>()) {
    // e.g. 2 != 3
    return irpass::analysis::same_statements(val1, val2)
               ? AliasResult::same
               : AliasResult::different;
  }
  if (auto bin1 = val1->cast<BinaryOpStmt>()) {
    if (auto bin2 = val2->cast<BinaryOpStmt>()) {
      if (bin1->lhs == bin2->lhs) {
        if (bin1->op_type == bin2->op_type) {
          if (bin1->rhs == bin2->rhs) {
            return AliasResult::same;
          } else if (bin1->rhs->is<ConstStmt>() && bin2->rhs->is<ConstStmt>()) {
            if (irpass::analysis::same_statements(bin1->rhs, bin2->rhs)) {
              return AliasResult::same;
            } else if (bin1->op_type == BinaryOpType::add ||
                       bin1->op_type == BinaryOpType::sub) {
              // e.g. x + 2 != x + 3
              return AliasResult::different;
            }
          }
        }
        return AliasResult::uncertain;
      }
    }
    if (bin1->lhs == val2 && bin1->rhs->is<ConstStmt>() &&
        (bin1->op_type == BinaryOpType::add ||
         bin1->op_type == BinaryOpType::sub) &&
        !bin1->rhs->as<ConstStmt>()->val[0].equal_value(0)) {
      // e.g. x + 2 != x
      return AliasResult::different;
    }
  }
  if (auto bin2 = val2->cast<BinaryOpStmt>()) {
    if (bin2->lhs == val1 && bin2->rhs->is<ConstStmt>() &&
        (bin2->op_type == BinaryOpType::add ||
         bin2->op_type == BinaryOpType::sub) &&
        !bin2->rhs->as<ConstStmt>()->val[0].equal_value(0)) {
      // e.g. x != x + 2
      return AliasResult::different;
    }
  }
  return AliasResult::uncertain;
}

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
  if (var1->is<StackAllocaStmt>() || var2->is<StackAllocaStmt>())
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
    if (auto ptr = s->cast<GlobalPtrStmt>())
      return ptr->snodes[0]->id;
    else if (auto get_child = s->cast<GetChStmt>())
      return get_child->output_snode->id;
    else
      return -1;
  };
  int snode1 = get_snode_id(var1);
  int snode2 = get_snode_id(var2);
  if (snode1 != -1 && snode2 != -1 && snode1 != snode2)
    return AliasResult::different;

  // GlobalPtrStmts with guaranteed different indices cannot share the same
  // address.
  if (var1->is<GlobalPtrStmt>() && var2->is<GlobalPtrStmt>()) {
    auto ptr1 = var1->as<GlobalPtrStmt>();
    auto ptr2 = var2->as<GlobalPtrStmt>();
    bool uncertain = false;
    for (int i = 0; i < (int)ptr1->indices.size(); i++) {
      auto current_result = same_value(ptr1->indices[i], ptr2->indices[i]);
      if (current_result == AliasResult::different)
        return AliasResult::different;
      else if (current_result == AliasResult::uncertain)
        uncertain = true;
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
