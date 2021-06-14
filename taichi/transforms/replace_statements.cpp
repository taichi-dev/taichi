#include "taichi/ir/transforms.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {

bool replace_and_insert_statements(
    IRNode *root,
    std::function<bool(Stmt *)> filter,
    std::function<std::unique_ptr<Stmt>(Stmt *)> generator) {
  return transform_statements(root, std::move(filter),
                              [&](Stmt *stmt, DelayedIRModifier *modifier) {
                                modifier->replace_with(stmt, generator(stmt));
                              });
}

bool replace_statements(IRNode *root,
                        std::function<bool(Stmt *)> filter,
                        std::function<Stmt *(Stmt *)> finder) {
  return transform_statements(
      root, std::move(filter), [&](Stmt *stmt, DelayedIRModifier *modifier) {
        auto existing_new_stmt = finder(stmt);
        irpass::replace_all_usages_with(root, stmt, existing_new_stmt);
        modifier->erase(stmt);
      });
}

}  // namespace irpass

TLANG_NAMESPACE_END
