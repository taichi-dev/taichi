#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

namespace irpass::analysis {

stmt_refs get_load_pointers(Stmt *load_stmt) {
  if (auto load_trait = load_stmt->cast<ir_traits::Load>()) {
    // The statement has the "Load" IR Trait
    return load_trait->get_load_pointers();
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

stmt_refs get_store_destination(Stmt *store_stmt) noexcept {
  // If store_stmt provides some data sources, return the pointers of the data.
  if (auto store_trait = store_stmt->cast<ir_traits::Store>()) {
    // The statement has the "Store" IR Trait
    return store_trait->get_store_destination();
  } else {
    return nullptr;
  }
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
