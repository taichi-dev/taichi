#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

namespace irpass::analysis {

// If there's TensorType involved,
// then return dest together with the aliased stmts
stmt_refs include_aliased_stmts(stmt_refs dest) {
  if (dest.size() == 1) {
    Stmt *dest_stmt = dest.begin()[0];
    if (dest_stmt->is<MatrixOfMatrixPtrStmt>()) {
      std::vector<Stmt *> rets = {dest_stmt};
      for (auto stmt : dest_stmt->as<MatrixOfMatrixPtrStmt>()->stmts) {
        if (stmt->is<MatrixPtrStmt>()) {
          rets.push_back(stmt);
          rets.push_back(stmt->as<MatrixPtrStmt>()->origin);
        }
      }
      return rets;
    }

    if (dest_stmt->is<MatrixPtrStmt>()) {
      std::vector<Stmt *> rets = {dest_stmt,
                                  dest_stmt->as<MatrixPtrStmt>()->origin};
      return rets;
    }
  }
  return dest;
}

stmt_refs get_load_pointers(Stmt *load_stmt, bool get_aliased) {
  if (auto load_trait = load_stmt->cast<ir_traits::Load>()) {
    // The statement has the "Load" IR Trait
    stmt_refs load_src = load_trait->get_load_pointers();
    if (get_aliased) {
      auto aliased_stmts = include_aliased_stmts(load_src);
      return aliased_stmts;
    }
    return load_src;
  }
  return nullptr;
}

Stmt *get_store_data(Stmt *store_stmt) noexcept {
  if (auto store_trait = store_stmt->cast<ir_traits::Store>()) {
    // The statement has the "Store" IR Trait
    return store_trait->get_store_data();
  }
  return nullptr;
}

stmt_refs get_store_destination(Stmt *store_stmt, bool get_aliased) noexcept {
  // If store_stmt provides some data sources, return the pointers of the data.
  if (auto store_trait = store_stmt->cast<ir_traits::Store>()) {
    // The statement has the "Store" IR Trait
    stmt_refs store_dest = store_trait->get_store_destination();
    if (get_aliased) {
      auto aliased_stmts = include_aliased_stmts(store_dest);
      return aliased_stmts;
    }
    return store_dest;
  } else {
    return nullptr;
  }
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
