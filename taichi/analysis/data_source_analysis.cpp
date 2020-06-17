#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {

Stmt *get_data_source(Stmt *store_stmt) {
  // If store_stmt provides a data source, return the data.
  // For convenience, return store_stmt if store_stmt is an AllocaStmt.
  if (store_stmt->is<AllocaStmt>()) {
    return store_stmt;
  } else if (auto local_store = store_stmt->cast<LocalStoreStmt>()) {
    return local_store->data;
  } else if (auto global_store = store_stmt->cast<GlobalStoreStmt>()) {
    return global_store->data;
  } else {
    return nullptr;
  }
}

Stmt *get_data_source_pointer(Stmt *store_stmt) {
  // If store_stmt provides a data source, return the pointer of the data.
  if (store_stmt->is<AllocaStmt>() || store_stmt->is<GlobalTemporaryStmt>() ||
      store_stmt->is<GlobalPtrStmt>() || store_stmt->is<ExternalPtrStmt>()) {
    // The statement itself provides a data source.
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

}

TLANG_NAMESPACE_END
