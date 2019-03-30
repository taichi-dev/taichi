#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <Eigen/Dense>

TLANG_NAMESPACE_BEGIN

#define TC_EXPRESSION_IMPLEMENTATION
#include "expression.h"

DecoratorRecorder dec;

IRBuilder &current_ast_builder() {
  return context->builder();
}

IRBuilder::ScopeGuard IRBuilder::create_scope(std::unique_ptr<Block> &list) {
  TC_ASSERT(list == nullptr);
  list = std::make_unique<Block>();
  if (!stack.empty()) {
    list->parent = stack.back();
  }
  return ScopeGuard(this, list.get());
}

void Expr::operator=(const Expr &o) {
  if (expr == nullptr) {
    set(o.eval());
  } else if (expr->is_lvalue()) {
    current_ast_builder().insert(
        std::make_unique<FrontendAssignStmt>(*this, load_if_ptr(o)));
  } else {
    // set(o.eval());
    TC_ERROR("Cannot assign to non-lvalue: {}", serialize());
  }
}

FrontendContext::FrontendContext() {
  root_node = std::make_unique<Block>();
  current_builder = std::make_unique<IRBuilder>(root_node.get());
}

Expr::Expr(int32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(Identifier id) : Expr() {
  expr = std::make_shared<IdExpression>(id);
}

Expr Expr::eval() const {
  TC_ASSERT(expr != nullptr);
  if (is<EvalExpression>()) {
    return *this;
  }
  auto eval_stmt = Stmt::make<FrontendEvalStmt>(*this);
  auto eval_expr = Expr::make<EvalExpression>(eval_stmt.get());
  eval_stmt->as<FrontendEvalStmt>()->eval_expr.set(eval_expr);
  // needed in lower_ast to replace the statement itself with the
  // lowered statement
  current_ast_builder().insert(std::move(eval_stmt));
  return eval_expr;
}

void Expr::operator+=(const Expr &o) {
  if (this->atomic) {
    current_ast_builder().insert(
        Stmt::make<FrontendAtomicStmt>(AtomicType::add, *this, o));
  } else {
    (*this) = (*this) + o;
  }
}
void Expr::operator-=(const Expr &o) {
  if (this->atomic) {
    current_ast_builder().insert(
        Stmt::make<FrontendAtomicStmt>(AtomicType::add, *this, -o));
  } else {
    (*this) = (*this) - o;
  }
}
void Expr::operator*=(const Expr &o) {
  TC_ASSERT(!this->atomic);
  (*this) = (*this) * o;
}
void Expr::operator/=(const Expr &o) {
  TC_ASSERT(!this->atomic);
  (*this) = (*this) / o;
}

FrontendForStmt::FrontendForStmt(ExpressionGroup loop_var, Expr global_var)
    : global_var(global_var) {
  vectorize = dec.vectorize;
  parallelize = dec.parallelize;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;

  loop_var_id.resize(loop_var.size());
  for (int i = 0; i < (int)loop_var.size(); i++) {
    loop_var_id[i] = loop_var[i].cast<IdExpression>()->id;
  }
}

FrontendForStmt::FrontendForStmt(ExpressionGroup loop_var, Expr begin, Expr end)
    : begin(begin), end(end) {
  vectorize = dec.vectorize;
  parallelize = dec.parallelize;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;
  loop_var_id.resize(loop_var.size());
  for (int i = 0; i < (int)loop_var.size(); i++) {
    loop_var_id[i] = loop_var[i].cast<IdExpression>()->id;
  }
}

IRNode *Stmt::get_ir_root() {
  auto block = parent;
  while (block->parent)
    block = block->parent;
  return dynamic_cast<IRNode *>(block);
}

FrontendAssignStmt::FrontendAssignStmt(Expr lhs, Expr rhs)
    : lhs(lhs), rhs(rhs) {
  TC_ASSERT(lhs->is_lvalue());
}

FrontendAtomicStmt::FrontendAtomicStmt(AtomicType op_type, Expr dest, Expr val)
    : op_type(op_type), dest(dest), val(val) {
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

int Identifier::id_counter = 0;
std::atomic<int> Stmt::instance_id_counter(0);

std::unique_ptr<FrontendContext> context;

void *Expr::evaluate_addr(int i, int j, int k, int l) {
  auto snode = this->cast<GlobalVariableExpression>()->snode;
  return snode->evaluate(get_current_program().data_structure, i, j, k, l);
}

template <int i, typename... Indices>
std::enable_if_t<(i < sizeof...(Indices)), int> get_if_exists(
    std::tuple<Indices...> tup) {
  static_assert(i >= 0, "i must be nonnegative");
  return std::get<i>(tup);
}

template <int i, typename... Indices>
std::enable_if_t<!(i < sizeof...(Indices)), int> get_if_exists(
    std::tuple<Indices...> tup) {
  static_assert(i >= 0, "i must be nonnegative");
  return 0;
}

template <typename... Indices>
void *Expr::val_tmp(DataType dt, Indices... indices) {
  auto snode = this->cast<GlobalVariableExpression>()->snode;
  if (dt != snode->dt) {
    TC_ERROR("Cannot access type {} as type {}", data_type_name(snode->dt),
             data_type_name(dt));
  }
  TC_ASSERT(sizeof...(indices) == snode->num_active_indices);
  int ind[max_num_indices];
  std::memset(ind, 0, sizeof(ind));
  auto tup = std::make_tuple(indices...);
#define LOAD_IND(i)                           \
  if (snode->physical_index_position[i] >= 0) \
    ind[snode->physical_index_position[i]] = get_if_exists<i>(tup);
  LOAD_IND(0);
  LOAD_IND(1);
  LOAD_IND(2);
  LOAD_IND(3);
#undef LOAD_IND
  TC_ASSERT(max_num_indices == 4);
  return evaluate_addr(ind[0], ind[1], ind[2], ind[3]);
}

template void *Expr::val_tmp<>(DataType);
template void *Expr::val_tmp<int>(DataType, int);
template void *Expr::val_tmp<int, int>(DataType, int, int);
template void *Expr::val_tmp<int, int, int>(DataType, int, int, int);
template void *Expr::val_tmp<int, int, int, int>(DataType, int, int, int, int);

void Stmt::insert_before_me(std::unique_ptr<Stmt> &&new_stmt) {
  TC_ASSERT(parent);
  auto &stmts = parent->statements;
  int loc = -1;
  for (int i = 0; i < (int)stmts.size(); i++) {
    if (stmts[i].get() == this) {
      loc = i;
      break;
    }
  }
  TC_ASSERT(loc != -1);
  new_stmt->parent = parent;
  stmts.insert(stmts.begin() + loc, std::move(new_stmt));
}

void Stmt::insert_after_me(std::unique_ptr<Stmt> &&new_stmt) {
  TC_ASSERT(parent);
  auto &stmts = parent->statements;
  int loc = -1;
  for (int i = 0; i < (int)stmts.size(); i++) {
    if (stmts[i].get() == this) {
      loc = i;
      break;
    }
  }
  TC_ASSERT(loc != -1);
  new_stmt->parent = parent;
  stmts.insert(stmts.begin() + loc + 1, std::move(new_stmt));
}

void Stmt::replace_with(Stmt *new_stmt) {
  auto root = get_ir_root();
  irpass::replace_all_usages_with(root, this, new_stmt);
  // Note: the current structure should have been destroyed now..
}

void Stmt::replace_operand_with(Stmt *old_stmt, Stmt *new_stmt) {
  for (int i = 0; i < num_operands(); i++) {
    if (operand(i) == old_stmt) {
      operand(i) = new_stmt;
    }
  }
}

Block *current_block = nullptr;

template <>
std::string to_string(const LaneAttribute<LocalAddress> &ptr) {
  std::string ret = " [";
  for (int i = 0; i < (int)ptr.size(); i++) {
    ret += fmt::format("{}[{}]", ptr[i].var->name(), ptr[i].offset);
    if (i + 1 < (int)ptr.size())
      ret += ", ";
  }
  ret += "]";
  return ret;
}

Stmt *LocalLoadStmt::previous_store_or_alloca_in_block() {
  int position = parent->locate(this);
  // TC_ASSERT(width() == 1);
  // TC_ASSERT(this->ptr[0].offset == 0);
  for (int i = position - 1; i >= 0; i--) {
    if (parent->statements[i]->is<LocalStoreStmt>()) {
      auto store = parent->statements[i]->as<LocalStoreStmt>();
      // TC_ASSERT(store->width() == 1);
      if (store->ptr == this->ptr[0].var) {
        // found
        return store;
      }
    } else if (parent->statements[i]->is<AllocaStmt>()) {
      auto alloca = parent->statements[i]->as<AllocaStmt>();
      // TC_ASSERT(alloca->width() == 1);
      if (alloca == this->ptr[0].var) {
        return alloca;
      }
    }
  }
  return nullptr;
}

TLANG_NAMESPACE_END

void taichi::Tlang::Block::erase(int location) {
  trash_bin.push_back(std::move(statements[location]));  // do not delete the
  // stmt, otherwise
  // print_ir will not
  // function properly
  statements.erase(statements.begin() + location);
}

void taichi::Tlang::Block::insert(std::unique_ptr<taichi::Tlang::Stmt> &&stmt,
                                  int location) {
  stmt->parent = this;
  if (location == -1) {
    statements.push_back(std::move(stmt));
  } else {
    statements.insert(statements.begin() + location, std::move(stmt));
  }
}

void taichi::Tlang::Block::replace_statements_in_range(
    int start,
    int end,
    taichi::Tlang::VecStatement &&stmts) {
  TC_ASSERT(start <= end);
  for (int i = 0; i < end - start; i++) {
    erase(start);
  }

  for (int i = 0; i < (int)stmts.size(); i++) {
    insert(std::move(stmts[i]), start + i);
  }
}

void taichi::Tlang::Block::replace_with(
    taichi::Tlang::Stmt *old_statement,
    std::unique_ptr<taichi::Tlang::Stmt> &&new_statement) {
  VecStatement vec;
  vec.push_back(std::move(new_statement));
  replace_with(old_statement, vec);
}

taichi::Tlang::Stmt *taichi::Tlang::Block::lookup_var(
    taichi::Tlang::Ident ident) const {
  auto ptr = local_var_alloca.find(ident);
  if (ptr != local_var_alloca.end()) {
    return ptr->second;
  } else {
    if (parent) {
      return parent->lookup_var(ident);
    } else {
      return nullptr;
    }
  }
}

taichi::Tlang::Stmt *taichi::Tlang::Block::mask() {
  if (mask_var)
    return mask_var;
  else if (parent == nullptr) {
    return nullptr;
  } else {
    return parent->mask();
  }
}
