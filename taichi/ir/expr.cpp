#include "taichi/ir/expr.h"

#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

void Expr::serialize(std::ostream &ss) const {
  TI_ASSERT(expr);
  expr->serialize(ss);
}

std::string Expr::serialize() const {
  std::stringstream ss;
  serialize(ss);
  return ss.str();
}

void Expr::set_tb(const std::string &tb) {
  expr->tb = tb;
}

void Expr::set_attribute(const std::string &key, const std::string &value) {
  expr->set_attribute(key, value);
}

std::string Expr::get_attribute(const std::string &key) const {
  return expr->get_attribute(key);
}

Expr select(const Expr &cond, const Expr &true_val, const Expr &false_val) {
  return Expr::make<TernaryOpExpression>(TernaryOpType::select, cond, true_val,
                                         false_val);
}

Expr operator-(const Expr &expr) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::neg, expr);
}

Expr operator~(const Expr &expr) {
  return Expr::make<UnaryOpExpression>(UnaryOpType::bit_not, expr);
}

Expr cast(const Expr &input, DataType dt) {
  auto ret =
      std::make_shared<UnaryOpExpression>(UnaryOpType::cast_value, input);
  ret->cast_type = dt;
  return Expr(ret);
}

Expr bit_cast(const Expr &input, DataType dt) {
  auto ret = std::make_shared<UnaryOpExpression>(UnaryOpType::cast_bits, input);
  ret->cast_type = dt;
  return Expr(ret);
}

Expr Expr::operator[](const ExprGroup &indices) const {
  TI_ASSERT(is<GlobalVariableExpression>() || is<ExternalTensorExpression>());
  return Expr::make<GlobalPtrExpression>(*this, indices.loaded());
}

Expr &Expr::operator=(const Expr &o) {
  if (get_current_program().current_callable) {
    // Inside a kernel or a function
    // Create an assignment in the IR
    if (expr == nullptr) {
      set(o.eval());
    } else if (expr->is_lvalue()) {
      current_ast_builder().insert(
          std::make_unique<FrontendAssignStmt>(*this, load_if_ptr(o)));
    } else {
      // set(o.eval());
      TI_ERROR("Cannot assign to non-lvalue: {}", serialize());
    }
  } else {
    set(o);  // Literally set this Expr to o
  }
  return *this;
}

Expr Expr::parent() const {
  TI_ASSERT_INFO(is<GlobalVariableExpression>(),
                 "Cannot get snode parent of non-global variables.");
  return Expr::make<GlobalVariableExpression>(
      cast<GlobalVariableExpression>()->snode->parent);
}

SNode *Expr::snode() const {
  TI_ASSERT_INFO(is<GlobalVariableExpression>(),
                 "Cannot get snode of non-global variables.");
  return cast<GlobalVariableExpression>()->snode;
}

Expr Expr::operator!() {
  return Expr::make<UnaryOpExpression>(UnaryOpType::logic_not, expr);
}

void Expr::declare(DataType dt) {
  set(Expr::make<GlobalVariableExpression>(dt, Identifier()));
}

void Expr::set_grad(const Expr &o) {
  this->cast<GlobalVariableExpression>()->adjoint.set(o);
}

Expr::Expr(int32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(int64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float32 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(float64 x) : Expr() {
  expr = std::make_shared<ConstExpression>(x);
}

Expr::Expr(const Identifier &id) : Expr() {
  expr = std::make_shared<IdExpression>(id);
}

Expr Expr::eval() const {
  TI_ASSERT(expr != nullptr);
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
    (*this) = Expr::make<AtomicOpExpression>(AtomicOpType::add, *this,
                                             load_if_ptr(o));
  } else {
    (*this) = (*this) + o;
  }
}

void Expr::operator-=(const Expr &o) {
  if (this->atomic) {
    (*this) = Expr::make<AtomicOpExpression>(AtomicOpType::sub, *this,
                                             load_if_ptr(o));
  } else {
    (*this) = (*this) - o;
  }
}

void Expr::operator*=(const Expr &o) {
  TI_ASSERT(!this->atomic);
  (*this) = (*this) * load_if_ptr(o);
}

void Expr::operator/=(const Expr &o) {
  TI_ASSERT(!this->atomic);
  (*this) = (*this) / load_if_ptr(o);
}

Expr load_if_ptr(const Expr &ptr) {
  if (ptr.is<GlobalPtrExpression>()) {
    return Expr::make<GlobalLoadExpression>(ptr);
  } else if (ptr.is<GlobalVariableExpression>()) {
    TI_ASSERT(ptr.cast<GlobalVariableExpression>()->snode->num_active_indices ==
              0);
    return Expr::make<GlobalLoadExpression>(ptr[ExprGroup()]);
  } else if (ptr.is<TensorElementExpression>()) {
    auto tensor_ptr = ptr.cast<TensorElementExpression>();
    if (tensor_ptr->is_global_tensor())
      return Expr::make<GlobalLoadExpression>(ptr);
    else if (tensor_ptr->is_local_tensor())
      return Expr::make<LocalLoadExpression>(ptr);
    else {
      TI_NOT_IMPLEMENTED
    }
  } else
    return ptr;
}

Expr Var(const Expr &x) {
  auto var = Expr(std::make_shared<IdExpression>());
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(var.expr)->id,
      PrimitiveType::unknown));
  var = x;
  return var;
}

TLANG_NAMESPACE_END
