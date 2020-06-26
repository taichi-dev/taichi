#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {

std::vector<Stmt *> get_load_pointers(Stmt *load_stmt) {
  // If load_stmt loads some variables or a stack, return the pointers of them.
  if (auto local_load = load_stmt->cast<LocalLoadStmt>()) {
    std::vector<Stmt *> result;
    for (auto &address : local_load->ptr.data) {
      if (std::find(result.begin(), result.end(), address.var) == result.end())
        result.push_back(address.var);
    }
    return result;
  } else if (auto global_load = load_stmt->cast<GlobalLoadStmt>()) {
    return std::vector<Stmt *>(1, global_load->ptr);
  } else if (auto atomic = load_stmt->cast<AtomicOpStmt>()) {
    return std::vector<Stmt *>(1, atomic->dest);
  } else if (auto stack_load_top = load_stmt->cast<StackLoadTopStmt>()) {
    return std::vector<Stmt *>(1, stack_load_top->stack);
  } else if (auto stack_load_top_adj = load_stmt->cast<StackLoadTopAdjStmt>()) {
    return std::vector<Stmt *>(1, stack_load_top_adj->stack);
  } else if (auto stack_acc_adj = load_stmt->cast<StackAccAdjointStmt>()) {
    // This statement loads and stores the adjoint data.
    return std::vector<Stmt *>(1, stack_acc_adj->stack);
  } else {
    return std::vector<Stmt *>();
  }
}

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
