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
  id = lhs.cast<IdExpression>()->id;
}

IRNode *FrontendContext::root() {
  return static_cast<IRNode *>(root_node.get());
}
int Identifier::id_counter = 0;
int Statement::id_counter = 0;

std::unique_ptr<FrontendContext> context;

TLANG_NAMESPACE_END
