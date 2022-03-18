#include <functional>
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expression_ops.h"

TLANG_NAMESPACE_BEGIN

template <typename Tf, typename Ti>
std::tuple<Expr, Expr, Expr> eig_3x3_export(ASTBuilder *ast_builder,
                                            const Expr &a00,
                                            const Expr &a01,
                                            const Expr &a02,
                                            const Expr &a10,
                                            const Expr &a11,
                                            const Expr &a12,
                                            const Expr &a20,
                                            const Expr &a21,
                                            const Expr &a22) {
  static_assert(sizeof(Tf) == sizeof(Ti), "");
  constexpr Tf M_SQRT3 = 1.73205080756887729352744634151;
  auto Var =
      std::bind(&ASTBuilder::make_var, ast_builder, std::placeholders::_1);

  auto A00 = Var(Expr(Tf(0.0)));
  auto A01 = Var(Expr(Tf(0.0)));
  auto A02 = Var(Expr(Tf(0.0)));
  auto A10 = Var(Expr(Tf(0.0)));
  auto A11 = Var(Expr(Tf(0.0)));
  auto A12 = Var(Expr(Tf(0.0)));
  auto A20 = Var(Expr(Tf(0.0)));
  auto A21 = Var(Expr(Tf(0.0)));
  auto A22 = Var(Expr(Tf(0.0)));
  ast_builder->insert_assignment(A00, a00);
  ast_builder->insert_assignment(A01, a01);
  ast_builder->insert_assignment(A02, a02);
  ast_builder->insert_assignment(A10, a10);
  ast_builder->insert_assignment(A11, a11);
  ast_builder->insert_assignment(A12, a12);
  ast_builder->insert_assignment(A20, a20);
  ast_builder->insert_assignment(A21, a21);
  ast_builder->insert_assignment(A22, a22);

  auto w0 = Var(Expr(Tf(0.0)));
  auto w1 = Var(Expr(Tf(0.0)));
  auto w2 = Var(Expr(Tf(0.0)));

  auto m = Var(Expr(Tf(0.0)));
  auto c1 = Var(Expr(Tf(0.0)));
  auto c0 = Var(Expr(Tf(0.0)));
  auto p = Var(Expr(Tf(0.0)));
  auto sqrt_p = Var(Expr(Tf(0.0)));
  auto q = Var(Expr(Tf(0.0)));
  auto c = Var(Expr(Tf(0.0)));
  auto s = Var(Expr(Tf(0.0)));
  auto phi = Var(Expr(Tf(0.0)));
  auto de = Var(Expr(Tf(0.0)));
  auto dd = Var(Expr(Tf(0.0)));
  auto ee = Var(Expr(Tf(0.0)));
  auto ff = Var(Expr(Tf(0.0)));

  ast_builder->insert_assignment(de, A01 * A12);
  ast_builder->insert_assignment(dd, A01 * A01);
  ast_builder->insert_assignment(ee, A12 * A12);
  ast_builder->insert_assignment(ff, A02 * A02);
  ast_builder->insert_assignment(m, A00 + A11 + A22);
  ast_builder->insert_assignment(
      c1, A00 * A11 + A00 * A22 + A11 * A22 - (dd + ee + ff));
  ast_builder->insert_assignment(c0, A22 * dd + A00 * ee + A11 * ff -
                                         A00 * A11 * A22 -
                                         Expr(Tf(2.0)) * A02 * de);
  ast_builder->insert_assignment(p, m * m - Expr(Tf(3.0)) * c1);
  ast_builder->insert_assignment(
      q, m * (p - Expr(Tf(1.5)) * c1) - Expr(Tf(13.5)) * c0);
  ast_builder->insert_assignment(sqrt_p, sqrt(abs(p)));
  ast_builder->insert_assignment(
      phi, Expr(Tf(27.0)) * (Expr(Tf(0.25)) * c1 * c1 * (p - c1) +
                             c0 * (q + Expr(Tf(6.75)) * c0)));
  ast_builder->insert_assignment(
      phi, Expr(Tf(1.0)) / Expr(Tf(3.0)) * atan2(sqrt(abs(phi)), q));
  ast_builder->insert_assignment(c, sqrt_p * cos(phi));
  ast_builder->insert_assignment(
      s, Expr(Tf(1.0)) / Expr(M_SQRT3) * sqrt_p * sin(phi));

  ast_builder->insert_assignment(w2, Expr(Tf(1.0)) / Expr(Tf(3.0)) * (m - c));
  ast_builder->insert_assignment(w1, w2 + s);
  ast_builder->insert_assignment(w0, w2 + c);
  ast_builder->insert_assignment(w2, w2 - s);
  return std::make_tuple(w0, w1, w2);
}

TLANG_NAMESPACE_END
