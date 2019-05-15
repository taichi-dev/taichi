#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
namespace taichi {
namespace math {
inline int maximum(int a) {
  return a;
}
}  // namespace math
}  // namespace taichi
#include <taichi/math.h>
#include <set>
#include <dlfcn.h>

#include "util.h"
#include "math.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

inline void layout(const std::function<void()> &body) {
  get_current_program().layout(body);
}

inline Program::Kernel &kernel(const std::function<void()> &body) {
  return get_current_program().kernel(body);
}

inline void kernel_name(std::string name) {
  get_current_program().get_current_kernel().name = name;
}

template <typename T>
inline void declare_var(Expr &a) {
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, get_data_type<T>()));
}

inline void declare_unnamed_var(Expr &a, DataType dt) {
  auto id = Identifier();
  auto a_ = Expr::make<IdExpression>(id);

  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(id, dt));

  if (a.expr) {
    a_ = a;  // assign
  }

  a.set(a_);
}

inline void declare_var(Expr &a) {
  current_ast_builder().insert(std::make_unique<FrontendAllocaStmt>(
      std::static_pointer_cast<IdExpression>(a.expr)->id, DataType::unknown));
}

#define Declare(x) auto x = Expr(std::make_shared<IdExpression>(#x));
#define DeclareNamed(x, name) \
  auto x = Expr(std::make_shared<IdExpression>(name));

#define NamedScalar(x, name, dt)   \
  DeclareNamed(x##_global, #name); \
  auto x = global_new(x##_global, DataType::dt);

#define Global(x, dt)  \
  Declare(x##_global); \
  auto x = global_new(x##_global, DataType::dt);

#define AmbientGlobal(x, dt, ambient)            \
  Declare(x##_global);                           \
  auto x = global_new(x##_global, DataType::dt); \
  set_ambient(x, ambient);

inline void set_ambient(Expr expr_, float32 val) {
  auto expr = expr_.cast<GlobalVariableExpression>();
  expr->ambient_value = TypedConstant(val);
  expr->has_ambient = true;
}

inline void set_ambient(Expr expr_, int32 val) {
  auto expr = expr_.cast<GlobalVariableExpression>();
  expr->ambient_value = TypedConstant(val);
  expr->has_ambient = true;
}

inline Expr global_new(Expr id_expr, DataType dt) {
  TC_ASSERT(id_expr.is<IdExpression>());
  auto ret = Expr(std::make_shared<GlobalVariableExpression>(
      dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

inline Expr global_new(DataType dt, std::string name = "") {
  auto id_expr = std::make_shared<IdExpression>(name);
  return Expr::make<GlobalVariableExpression>(dt, id_expr->id);
}

template <typename T>
inline Expr Rand() {
  return Expr::make<RandExpression>(get_data_type<T>());
}

template <typename T>
inline T Eval(const T &t) {
  return t.eval();
}

inline Expr copy(const Expr &expr) {
  auto e = expr.eval();
  auto stmt = Stmt::make<ElementShuffleStmt>(
      VectorElement(e.cast<EvalExpression>()->stmt_ptr, 0));
  auto eval_expr = std::make_shared<EvalExpression>(stmt.get());
  current_ast_builder().insert(std::move(stmt));
  return Expr(eval_expr);
}

template <typename... indices>
std::vector<Index> Indices(indices... ind) {
  auto ind_vec = std::vector<int>({ind...});
  std::vector<Index> ret;
  for (auto in : ind_vec) {
    ret.push_back(Index(in));
  }
  return ret;
}

inline Expr Atomic(Expr dest) {
  // NOTE: dest must be passed by value so that the original
  // expr will not be modified into an atomic one.
  dest.atomic = true;
  return dest;
}

// expr_group are indices
inline void Activate(SNode *snode, const ExpressionGroup &expr_group) {
  current_ast_builder().insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::activate, snode, expr_group));
}

inline void Activate(const Expr &expr, const ExpressionGroup &expr_group) {
  return Activate(expr.snode(), expr_group);
}

inline void Deactivate(SNode *snode, const ExpressionGroup &expr_group) {
  current_ast_builder().insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::deactivate, snode, expr_group));
}

inline void Append(SNode *snode,
                   const ExpressionGroup &indices,
                   const Expr &val) {
  current_ast_builder().insert(Stmt::make<FrontendSNodeOpStmt>(
      SNodeOpType::append, snode, indices, val));
}

inline void Append(const Expr &expr,
                   const ExpressionGroup &indices,
                   const Expr &val) {
  Append(expr.snode(), indices, val);
}

inline void InsertAssert(const std::string &text, const Expr &expr) {
  current_ast_builder().insert(Stmt::make<FrontendAssertStmt>(text, expr));
}

inline void Clear(SNode *snode, const ExpressionGroup &indices) {
  current_ast_builder().insert(
      Stmt::make<FrontendSNodeOpStmt>(SNodeOpType::clear, snode, indices));
}

inline void Clear(const Expr &expr, const ExpressionGroup &indices) {
  return Clear(expr.snode(), indices);
}

inline Expr Probe(SNode *snode, const ExpressionGroup &indices) {
  return Expr::make<ProbeExpression>(snode, indices);
}

inline Expr Probe(const Expr &expr, const ExpressionGroup &indices) {
  return Probe(expr.snode(), indices);
}

inline Expr AssumeInRange(const Expr &expr,
                          const Expr &base,
                          int low,
                          int high) {
  return Expr::make<RangeAssumptionExpression>(expr, base, low, high);
}

inline void benchmark_kernel() {
  get_current_program().get_current_kernel().benchmarking = true;
}

#define Kernel(x) auto &x = get_current_program().kernel(#x)
#define Assert(x) InsertAssert(#x, (x))

TLANG_NAMESPACE_END

TC_NAMESPACE_BEGIN
inline Dict parse_param(std::vector<std::string> cli_param) {
  Dict dict;
  for (auto &s : cli_param) {
    auto div = s.find('=');
    if (div == std::string::npos) {
      TC_INFO("CLI parameter format: key=value, e.g. file_name=test.bin.");
      exit(-1);
    }
    dict.set(s.substr(0, div), s.substr(div + 1));
  }
  TC_P(dict);
  return dict;
}

void write_partio(std::vector<Vector3> positions, const std::string &file_name);
TC_NAMESPACE_END
