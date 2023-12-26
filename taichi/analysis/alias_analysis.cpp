#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

namespace irpass::analysis {

namespace {

/**
 * @brief Retrieve the local alloca statement from a given statement.
 *
 * @param var A pointer to the statement, which could be a MatrixPtrStmt, an
 * AllocaStmt, or else.
 * @return A pointer to the local alloca statement if found, else nullptr.
 */
Stmt *retrieve_local(Stmt *var) {
  if (var->is<AllocaStmt>()) {
    return var;
  } else if (var->is<MatrixPtrStmt>() &&
             var->cast<MatrixPtrStmt>()->offset_used_as_index()) {
    return var->cast<MatrixPtrStmt>()->origin;
  } else {
    return (Stmt *)nullptr;
  }
}

/**
 * @brief Get the SNode ID from a given statement.
 *
 * @param s A pointer to the statement, which could be a GlobalPtrStmt, a
 * GetChStmt, or else.
 * @return The ID of the SNode if found, else -1.
 */
int get_snode_id(Stmt *s) {
  if (auto ptr = s->cast<GlobalPtrStmt>()) {
    return ptr->snode->id;
  } else if (auto get_child = s->cast<GetChStmt>()) {
    return get_child->output_snode->id;
  }
  return -1;
};

}  // namespace

AliasResult alias_analysis(Stmt *var1, Stmt *var2) {
  // If both stmts are allocas, they have the same address iff var1 == var2.
  // If only one of them is an alloca, they can never share the same address.
  if (var1 == var2)
    return AliasResult::same;
  if (!var1 || !var2)
    return AliasResult::different;

  // Check aliasing based on ExternalTensorBasePtrStmt
  if (var1->is<ExternalTensorBasePtrStmt>() ||
      var2->is<ExternalTensorBasePtrStmt>()) {
    auto *base = var1->cast<ExternalTensorBasePtrStmt>();
    Stmt *other = var2;
    if (!base) {
      base = var2->cast<ExternalTensorBasePtrStmt>();
      other = var1;
    }

    // Test if `other` is an ExternalPtrStmt
    auto *external_ptr = other->cast<ExternalPtrStmt>();
    if (!external_ptr) {
      // `other` is not an ExternalPtrStmt, try to check whether it has an
      // ExternalPtrStmt as its origin If not, the pointers are not aliased
      if (auto *matrix_ptr = other->cast<MatrixPtrStmt>()) {
        external_ptr = matrix_ptr->origin->cast<ExternalPtrStmt>();
        if (!external_ptr)
          return AliasResult::different;
      }
    }
    // If one external pointer references the grad tensor and the other
    // references the data tensor, they are not aliased
    if (base->is_grad != external_ptr->is_grad)
      return AliasResult::different;
    // If both external pointers load from the same NDArray (same `arg_id`),
    // they could be aliased
    if (base->arg_id == external_ptr->base_ptr->as<ArgLoadStmt>()->arg_id) {
      return AliasResult::uncertain;
    }
    // If none of the above conditions are met, the pointers are not aliased
    return AliasResult::different;
  }

  Stmt *origin1 = retrieve_local(var1);
  Stmt *origin2 = retrieve_local(var2);
  if (origin1 != nullptr && origin2 != nullptr) {
    // If both statements comes from local allocas...
    if (var1->is<MatrixPtrStmt>() && var2->is<MatrixPtrStmt>()) {
      // If both statements come from the same local alloca,
      // they have the same address iff they have the same offset / index.
      if (origin1 == origin2 ||
          alias_analysis(origin1, origin2) == AliasResult::same) {
        auto diff = value_diff_ptr_index(var1->cast<MatrixPtrStmt>()->offset,
                                         var2->cast<MatrixPtrStmt>()->offset);
        if (diff.is_diff_certain) {
          return diff.diff_range == 0 ? AliasResult::same
                                      : AliasResult::different;
        }
      } else {
        return AliasResult::different;
      }
    }

    if (origin1 == origin2) {
      return AliasResult::uncertain;
    }

    if (origin1->is<AllocaStmt>() || origin2->is<AllocaStmt>())
      return AliasResult::different;

    // At this point we have checked `origin1` and `origin2` are:
    // - not both MatrixPtrStmt
    // - not the same Stmt
    // - neither one is AllocatorStmt
    // Thus they must be both `GlobalTemporaryStmt`.
    TI_ASSERT(origin1->is<GlobalTemporaryStmt>() &&
              origin2->is<GlobalTemporaryStmt>());
    if (origin1->cast<GlobalTemporaryStmt>()->offset ==
        origin2->cast<GlobalTemporaryStmt>()->offset) {
      return AliasResult::uncertain;
    } else {
      return AliasResult::different;
    }
  } else if (origin1 != nullptr || origin2 != nullptr) {
    // One comes from local alloca, the other doesn't, must be different.
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

  // Check based on ExternalPtrStmt
  if (var1->is<ExternalPtrStmt>() || var2->is<ExternalPtrStmt>()) {
    if (!var1->is<ExternalPtrStmt>() || !var2->is<ExternalPtrStmt>())
      return AliasResult::different;
    auto ptr1 = var1->as<ExternalPtrStmt>();
    auto ptr2 = var2->as<ExternalPtrStmt>();
    // If `var1` & `var2` loads from different NDarrays, or one loads the
    // gradient tensor while the other does not, they must be different
    if (ptr1->base_ptr != ptr2->base_ptr) {
      auto base1 = ptr1->base_ptr->as<ArgLoadStmt>();
      auto base2 = ptr2->base_ptr->as<ArgLoadStmt>();
      if (base1->arg_id != base2->arg_id || ptr1->is_grad != ptr2->is_grad) {
        return AliasResult::different;
      }
    } else if (ptr1->is_grad != ptr2->is_grad) {
      return AliasResult::different;
    }
    // If both loads from the same NDArray, they could be aliased, checking the
    // offset difference
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

}  // namespace taichi::lang
