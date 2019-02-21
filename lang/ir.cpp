#include "ir.h"
#include <numeric>
#include "tlang.h"
#include <Eigen/Dense>

TLANG_NAMESPACE_BEGIN

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

void ExprH::operator=(const ExpressionHandle &o) {
  current_ast_builder().insert(std::make_unique<AssignStmt>(*this, o));
}

FrontendContext::FrontendContext() {
  root_node = std::make_unique<Block>();
  current_builder = std::make_unique<IRBuilder>(root_node.get());
}

ExpressionHandle::ExpressionHandle(int x) {
  expr = std::make_shared<ConstExpression>(x);
}

ExpressionHandle::ExpressionHandle(double x) {
  expr = std::make_shared<ConstExpression>(x);
}

ExpressionHandle::ExpressionHandle(Identifier id) {
  expr = std::make_shared<IdExpression>(id);
}

FrontendForStmt::FrontendForStmt(ExprH loop_var, ExprH begin, ExprH end)
    : begin(begin), end(end) {
  loop_var_id = loop_var.cast<IdExpression>()->id;
}

AssignStmt::AssignStmt(ExprH lhs, ExprH rhs) : lhs(lhs), rhs(rhs) {
  TC_ASSERT(lhs.is<IdExpression>() || lhs.is<GlobalPtrExpression>());
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}

int Identifier::id_counter = 0;
int Statement::id_counter = 0;

std::unique_ptr<FrontendContext> context;

void *ExprH::evaluate_addr(int i, int j, int k, int l) {
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
void *ExprH::val_tmp(Indices... indices) {
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
  /*
  TC_P(snode->index_order[0]);
  TC_P(snode->index_order[1]);
  TC_P(snode->index_order[2]);
  TC_P(snode->index_order[3]);
  TC_P(ind[0]);
  TC_P(ind[1]);
  TC_P(ind[2]);
  TC_P(ind[3]);
  TC_P(((int *)&tup)[0]);
  TC_P(((int *)&tup)[1]);
  TC_P(((int *)&tup)[2]);
  TC_P(((int *)&tup)[3]);
  */
  return evaluate_addr(ind[0], ind[1], ind[2], ind[3]);
}

template void *ExprH::val_tmp<>();
template void *ExprH::val_tmp<int>(int);
template void *ExprH::val_tmp<int, int>(int, int);
template void *ExprH::val_tmp<int, int, int>(int, int, int);
template void *ExprH::val_tmp<int, int, int, int>(int, int, int, int);
TLANG_NAMESPACE_END
