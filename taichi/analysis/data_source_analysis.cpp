#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"

namespace taichi::lang {

namespace irpass::analysis {

stmt_refs get_load_pointers(Stmt *load_stmt) {
  if (auto load_trait = load_stmt->cast<IRTraits::Load>()) {
    // The statement has the "Load" IR Trait
    return load_trait->get_load_pointers();
  }
  return nullptr;
  /*
  // If load_stmt loads some variables or a stack, return the pointers of them.
  if (auto local_load = load_stmt->cast<LocalLoadStmt>()) {
    return std::vector<Stmt *>(1, local_load->src);
  } else if (auto global_load = load_stmt->cast<GlobalLoadStmt>()) {
    return std::vector<Stmt *>(1, global_load->src);
  } else if (auto atomic = load_stmt->cast<AtomicOpStmt>()) {
    return std::vector<Stmt *>(1, atomic->dest);
  } else if (auto stack_load_top = load_stmt->cast<AdStackLoadTopStmt>()) {
    return std::vector<Stmt *>(1, stack_load_top->stack);
  } else if (auto stack_load_top_adj =
                 load_stmt->cast<AdStackLoadTopAdjStmt>()) {
    return std::vector<Stmt *>(1, stack_load_top_adj->stack);
  } else if (auto stack_acc_adj = load_stmt->cast<AdStackAccAdjointStmt>()) {
    // This statement loads and stores the adjoint data.
    return std::vector<Stmt *>(1, stack_acc_adj->stack);
  } else if (auto stack_push = load_stmt->cast<AdStackPushStmt>()) {
    // This is to make dead store elimination not eliminate consequent pushes.
    return std::vector<Stmt *>(1, stack_push->stack);
  } else if (auto stack_pop = load_stmt->cast<AdStackPopStmt>()) {
    // This is to make dead store elimination not eliminate consequent pops.
    return std::vector<Stmt *>(1, stack_pop->stack);
  } else if (auto external_func = load_stmt->cast<ExternalFuncCallStmt>()) {
    return external_func->arg_stmts;
  } else if (auto ref = load_stmt->cast<ReferenceStmt>()) {
    return {ref->var};
  } else {
    return std::vector<Stmt *>();
  }
  */
}

Stmt *get_store_data(Stmt *store_stmt) noexcept {
  if (auto store_trait = store_stmt->cast<IRTraits::Store>()) {
    // The statement has the "Store" IR Trait
    return store_trait->get_store_data();
  }
  return nullptr;

  /*
  // If store_stmt provides one data source, return the data.
  if (store_stmt->is<AllocaStmt>() && !store_stmt->ret_type->is<TensorType>()) {
    // For convenience, return store_stmt instead of the const [0] it actually
    // stores.
    return store_stmt;
  } else if (auto local_store = store_stmt->cast<LocalStoreStmt>()) {
    return local_store->val;
  } else if (auto global_store = store_stmt->cast<GlobalStoreStmt>()) {
    return global_store->val;
  } else {
    return nullptr;
  }
  */
}

stmt_refs get_store_destination(Stmt *store_stmt) noexcept {
  // If store_stmt provides some data sources, return the pointers of the data.
  if (auto store_trait = store_stmt->cast<IRTraits::Store>()) {
    // The statement has the "Store" IR Trait
    return store_trait->get_store_destination();
  } else {
    return nullptr;
  }

  /*
  // If store_stmt provides some data sources, return the pointers of the data.
  if (store_stmt->is<AllocaStmt>() && !store_stmt->ret_type->is<TensorType>()) {
    // The statement itself provides a data source (const [0]).
    return std::vector<Stmt *>(1, store_stmt);
  } else if (auto local_store = store_stmt->cast<LocalStoreStmt>()) {
    return std::vector<Stmt *>(1, local_store->dest);
  } else if (auto global_store = store_stmt->cast<GlobalStoreStmt>()) {
    return std::vector<Stmt *>(1, global_store->dest);
  } else if (auto atomic = store_stmt->cast<AtomicOpStmt>()) {
    return std::vector<Stmt *>(1, atomic->dest);
  } else if (auto snode_op = store_stmt->cast<SNodeOpStmt>()) {
    if (snode_op->op_type == SNodeOpType::allocate) {
      return {snode_op->val, snode_op->ptr};
    } else {
      return {};
    }
  } else if (auto external_func = store_stmt->cast<ExternalFuncCallStmt>()) {
    if (store_stmt->cast<ExternalFuncCallStmt>()->type ==
        ExternalFuncCallStmt::BITCODE) {
      return external_func->arg_stmts;
    } else {
      return external_func->output_stmts;
    }
  } else {
    return std::vector<Stmt *>();
  }
  */
}

}  // namespace irpass::analysis

}  // namespace taichi::lang
