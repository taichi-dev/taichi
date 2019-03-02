#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <Eigen/Dense>

TLANG_NAMESPACE_BEGIN

DecoratorRecorder dec;

// Vector width, vectorization plan etc
class PropagateSchedule : public IRVisitor {};

IRBuilder::ScopeGuard IRBuilder::create_scope(std::unique_ptr<Block> &list) {
  TC_ASSERT(list == nullptr);
  list = std::make_unique<Block>();
  if (!stack.empty()) {
    list->parent = stack.back();
  }
  return ScopeGuard(this, list.get());
}

void Expr::operator=(const Expr &o) {
  if (this->expr == nullptr) {
    // create an anonymous local variable
    auto id = Identifier();
    expr = std::make_shared<IdExpression>(id);
    declare_var(*this);
    /*
    auto stmt = std::make_unique<FrontendTmpValStmt>(o);
    expr = std::make_shared<TmpValExpression>(stmt.get());
    current_ast_builder().insert(std::move(stmt));
    */
  }
  current_ast_builder().insert(std::make_unique<FrontendAssignStmt>(*this, o));
}

FrontendContext::FrontendContext() {
  root_node = std::make_unique<Block>();
  current_builder = std::make_unique<IRBuilder>(root_node.get());
}

Expr::Expr(int32 x) {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float32 x) {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(Identifier id) {
  expr = std::make_shared<IdExpression>(id);
}

void Expr::operator+=(const Expr &o) {
  (*this) = (*this) + o;
}
void Expr::operator-=(const Expr &o) {
  (*this) = (*this) - o;
}
void Expr::operator*=(const Expr &o) {
  (*this) = (*this) * o;
}
void Expr::operator/=(const Expr &o) {
  (*this) = (*this) / o;
}

FrontendForStmt::FrontendForStmt(Expr loop_var, Expr begin, Expr end)
    : begin(begin), end(end) {
  vectorize = dec.vectorize;
  parallelize = dec.parallelize;
  dec.reset();
  if (vectorize == -1)
    vectorize = 1;
  loop_var_id = loop_var.cast<IdExpression>()->id;
}

IRNode *Stmt::get_ir_root() {
  auto block = parent;
  while (block->parent)
    block = block->parent;
  return dynamic_cast<IRNode *>(block);
}

FrontendAssignStmt::FrontendAssignStmt(Expr lhs, Expr rhs)
    : lhs(lhs), rhs(rhs) {
  TC_ASSERT(lhs.is<IdExpression>() || lhs.is<GlobalPtrExpression>());
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

int Identifier::id_counter = 0;
int Statement::id_counter = 0;

std::unique_ptr<FrontendContext> context;

void *Expr::evaluate_addr(int i, int j, int k, int l) {
  auto snode = this->cast<GlobalVariableExpression>()->snode;
  return snode->evaluate(get_current_program().data_structure, i, j, k, l);
}

template <int i, typename... Indices>
std::enable_if_t<(i < sizeof...(Indices)), int> get_if_exists(
    std::tuple<Indices...> tup) {
  return std::get<i>(tup);
}

template <int i, typename... Indices>
std::enable_if_t<!(i < sizeof...(Indices)), int> get_if_exists(
    std::tuple<Indices...> tup) {
  return 0;
}

template <typename... Indices>
void *Expr::val_tmp(Indices... indices) {
  auto snode = this->cast<GlobalVariableExpression>()->snode;
  TC_ASSERT(sizeof...(indices) == snode->num_active_indices);
  int ind[max_num_indices];
  std::memset(ind, 0, sizeof(ind));
  auto tup = std::make_tuple(indices...);
#define LOAD_IND(i) ind[snode->index_order[i]] = get_if_exists<i>(tup);
  LOAD_IND(0);
  LOAD_IND(1);
  LOAD_IND(2);
  LOAD_IND(3);
#undef LOAD_IND
  TC_ASSERT(max_num_indices == 4);
  return evaluate_addr(ind[0], ind[1], ind[2], ind[3]);
}

template void *Expr::val_tmp<>();
template void *Expr::val_tmp<int>(int);
template void *Expr::val_tmp<int, int>(int, int);
template void *Expr::val_tmp<int, int, int>(int, int, int);
template void *Expr::val_tmp<int, int, int, int>(int, int, int, int);

void Stmt::insert_before(std::unique_ptr<Stmt> &&new_stmt) {
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

void Stmt::insert_after(std::unique_ptr<Stmt> &&new_stmt) {
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

Block *current_block = nullptr;

TLANG_NAMESPACE_END
