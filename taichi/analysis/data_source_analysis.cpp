#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {

Stmt *get_store_data(Stmt *store_stmt) {
  // If store_stmt provides a data source, return the data.
  if (store_stmt->is<AllocaStmt>()) {
    // For convenience, return store_stmt instead of the const [0] it actually
    // stores.
    return store_stmt;
  } else if (auto local_store = store_stmt->cast<LocalStoreStmt>()) {
    return local_store->data;
  } else if (auto global_store = store_stmt->cast<GlobalStoreStmt>()) {
    return global_store->data;
  } else {
    return nullptr;
  }
}

Stmt *get_store_destination(Stmt *store_stmt) {
  // If store_stmt provides a data source, return the pointer of the data.
  if (store_stmt->is<AllocaStmt>()) {
    // The statement itself provides a data source (const [0]).
    return store_stmt;
  } else if (auto local_store = store_stmt->cast<LocalStoreStmt>()) {
    return local_store->ptr;
  } else if (auto global_store = store_stmt->cast<GlobalStoreStmt>()) {
    return global_store->ptr;
  } else if (auto atomic = store_stmt->cast<AtomicOpStmt>()) {
    return atomic->dest;
  } else {
    return nullptr;
  }
}

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
