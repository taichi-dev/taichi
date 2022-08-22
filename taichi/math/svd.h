#include <functional>
#include "taichi/ir/frontend_ir.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/expression_ops.h"

TLANG_NAMESPACE_BEGIN

template <typename Tf, typename Ti>
Expr svd_bitwise_or(const Expr &a, const Expr &b) {
  return bit_cast<Tf>(bit_cast<Ti>(a) | bit_cast<Ti>(b));
}

template <typename Tf, typename Ti>
Expr svd_bitwise_xor(const Expr &a, const Expr &b) {
  return bit_cast<Tf>(bit_cast<Ti>(a) ^ bit_cast<Ti>(b));
}

template <typename Tf, typename Ti>
Expr svd_bitwise_and(const Expr &a, const Expr &b) {
  return bit_cast<Tf>(bit_cast<Ti>(a) & bit_cast<Ti>(b));
}

template <typename Tf, typename Ti>
std::tuple<Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr,
           Expr>
sifakis_svd_export(ASTBuilder *ast_builder,
                   const Expr &a00,
                   const Expr &a01,
                   const Expr &a02,
                   const Expr &a10,
                   const Expr &a11,
                   const Expr &a12,
                   const Expr &a20,
                   const Expr &a21,
                   const Expr &a22,
                   int num_iters) {
  static_assert(sizeof(Tf) == sizeof(Ti), "");
  constexpr Tf Four_Gamma_Squared = 5.82842712474619f;
  constexpr Tf Sine_Pi_Over_Eight = 0.3826834323650897f;
  constexpr Tf Cosine_Pi_Over_Eight = 0.9238795325112867f;

  std::string tb = "";
  auto Var =
      std::bind(&ASTBuilder::make_var, ast_builder, std::placeholders::_1, tb);

  auto Sfour_gamma_squared = Var(Expr(Tf(0.0)));
  auto Ssine_pi_over_eight = Var(Expr(Tf(0.0)));
  auto Scosine_pi_over_eight = Var(Expr(Tf(0.0)));
  auto Sone_half = Var(Expr(Tf(0.0)));
  auto Sone = Var(Expr(Tf(0.0)));
  auto Stiny_number = Var(Expr(Tf(0.0)));
  auto Ssmall_number = Var(Expr(Tf(0.0)));
  auto Sa11 = Var(Expr(Tf(0.0)));
  auto Sa21 = Var(Expr(Tf(0.0)));
  auto Sa31 = Var(Expr(Tf(0.0)));
  auto Sa12 = Var(Expr(Tf(0.0)));
  auto Sa22 = Var(Expr(Tf(0.0)));
  auto Sa32 = Var(Expr(Tf(0.0)));
  auto Sa13 = Var(Expr(Tf(0.0)));
  auto Sa23 = Var(Expr(Tf(0.0)));
  auto Sa33 = Var(Expr(Tf(0.0)));
  auto Sv11 = Var(Expr(Tf(0.0)));
  auto Sv21 = Var(Expr(Tf(0.0)));
  auto Sv31 = Var(Expr(Tf(0.0)));
  auto Sv12 = Var(Expr(Tf(0.0)));
  auto Sv22 = Var(Expr(Tf(0.0)));
  auto Sv32 = Var(Expr(Tf(0.0)));
  auto Sv13 = Var(Expr(Tf(0.0)));
  auto Sv23 = Var(Expr(Tf(0.0)));
  auto Sv33 = Var(Expr(Tf(0.0)));
  auto Su11 = Var(Expr(Tf(0.0)));
  auto Su21 = Var(Expr(Tf(0.0)));
  auto Su31 = Var(Expr(Tf(0.0)));
  auto Su12 = Var(Expr(Tf(0.0)));
  auto Su22 = Var(Expr(Tf(0.0)));
  auto Su32 = Var(Expr(Tf(0.0)));
  auto Su13 = Var(Expr(Tf(0.0)));
  auto Su23 = Var(Expr(Tf(0.0)));
  auto Su33 = Var(Expr(Tf(0.0)));
  auto Sc = Var(Expr(Tf(0.0)));
  auto Ss = Var(Expr(Tf(0.0)));
  auto Sch = Var(Expr(Tf(0.0)));
  auto Ssh = Var(Expr(Tf(0.0)));
  auto Stmp1 = Var(Expr(Tf(0.0)));
  auto Stmp2 = Var(Expr(Tf(0.0)));
  auto Stmp3 = Var(Expr(Tf(0.0)));
  auto Stmp4 = Var(Expr(Tf(0.0)));
  auto Stmp5 = Var(Expr(Tf(0.0)));
  auto Sqvs = Var(Expr(Tf(0.0)));
  auto Sqvvx = Var(Expr(Tf(0.0)));
  auto Sqvvy = Var(Expr(Tf(0.0)));
  auto Sqvvz = Var(Expr(Tf(0.0)));
  auto Ss11 = Var(Expr(Tf(0.0)));
  auto Ss21 = Var(Expr(Tf(0.0)));
  auto Ss31 = Var(Expr(Tf(0.0)));
  auto Ss22 = Var(Expr(Tf(0.0)));
  auto Ss32 = Var(Expr(Tf(0.0)));
  auto Ss33 = Var(Expr(Tf(0.0)));
  ast_builder->insert_assignment(Sfour_gamma_squared, Expr(Four_Gamma_Squared));
  ast_builder->insert_assignment(Ssine_pi_over_eight, Expr(Sine_Pi_Over_Eight));
  ast_builder->insert_assignment(Scosine_pi_over_eight,
                                 Expr(Cosine_Pi_Over_Eight));
  ast_builder->insert_assignment(Sone_half, Expr(Tf(0.5f)));
  ast_builder->insert_assignment(Sone, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Stiny_number, Expr(Tf(1.e-20f)));
  ast_builder->insert_assignment(Ssmall_number, Expr(Tf(1.e-12f)));
  ast_builder->insert_assignment(Sa11, a00);
  ast_builder->insert_assignment(Sa21, a10);
  ast_builder->insert_assignment(Sa31, a20);
  ast_builder->insert_assignment(Sa12, a01);
  ast_builder->insert_assignment(Sa22, a11);
  ast_builder->insert_assignment(Sa32, a21);
  ast_builder->insert_assignment(Sa13, a02);
  ast_builder->insert_assignment(Sa23, a12);
  ast_builder->insert_assignment(Sa33, a22);
  ast_builder->insert_assignment(Sqvs, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Sqvvx, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Sqvvy, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Sqvvz, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Ss11, Sa11 * Sa11);
  ast_builder->insert_assignment(Stmp1, Sa21 * Sa21);
  ast_builder->insert_assignment(Ss11, Stmp1 + Ss11);
  ast_builder->insert_assignment(Stmp1, Sa31 * Sa31);
  ast_builder->insert_assignment(Ss11, Stmp1 + Ss11);
  ast_builder->insert_assignment(Ss21, Sa12 * Sa11);
  ast_builder->insert_assignment(Stmp1, Sa22 * Sa21);
  ast_builder->insert_assignment(Ss21, Stmp1 + Ss21);
  ast_builder->insert_assignment(Stmp1, Sa32 * Sa31);
  ast_builder->insert_assignment(Ss21, Stmp1 + Ss21);
  ast_builder->insert_assignment(Ss31, Sa13 * Sa11);
  ast_builder->insert_assignment(Stmp1, Sa23 * Sa21);
  ast_builder->insert_assignment(Ss31, Stmp1 + Ss31);
  ast_builder->insert_assignment(Stmp1, Sa33 * Sa31);
  ast_builder->insert_assignment(Ss31, Stmp1 + Ss31);
  ast_builder->insert_assignment(Ss22, Sa12 * Sa12);
  ast_builder->insert_assignment(Stmp1, Sa22 * Sa22);
  ast_builder->insert_assignment(Ss22, Stmp1 + Ss22);
  ast_builder->insert_assignment(Stmp1, Sa32 * Sa32);
  ast_builder->insert_assignment(Ss22, Stmp1 + Ss22);
  ast_builder->insert_assignment(Ss32, Sa13 * Sa12);
  ast_builder->insert_assignment(Stmp1, Sa23 * Sa22);
  ast_builder->insert_assignment(Ss32, Stmp1 + Ss32);
  ast_builder->insert_assignment(Stmp1, Sa33 * Sa32);
  ast_builder->insert_assignment(Ss32, Stmp1 + Ss32);
  ast_builder->insert_assignment(Ss33, Sa13 * Sa13);
  ast_builder->insert_assignment(Stmp1, Sa23 * Sa23);
  ast_builder->insert_assignment(Ss33, Stmp1 + Ss33);
  ast_builder->insert_assignment(Stmp1, Sa33 * Sa33);
  ast_builder->insert_assignment(Ss33, Stmp1 + Ss33);
  ast_builder->strictly_serialize();
  ast_builder->insert_for(Expr(0), Expr(num_iters), [&](Expr sweep) {
    ast_builder->insert_assignment(Ssh, Ss21 * Sone_half);
    ast_builder->insert_assignment(Stmp5, Ss11 - Ss22);
    ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
    ast_builder->insert_assignment(
        Stmp1,
        bit_cast<Tf>(expr_select(Stmp2 >= Stiny_number,
                                 Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
    ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Stmp1, Ssh));
    ast_builder->insert_assignment(Sch, svd_bitwise_and<Tf, Ti>(Stmp1, Stmp5));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Sone));
    ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp2));
    ast_builder->insert_assignment(Stmp1, Ssh * Ssh);
    ast_builder->insert_assignment(Stmp2, Sch * Sch);
    ast_builder->insert_assignment(Stmp3, Stmp1 + Stmp2);
    ast_builder->insert_assignment(Stmp4, rsqrt(Stmp3));
    ast_builder->insert_assignment(Ssh, Stmp4 * Ssh);
    ast_builder->insert_assignment(Sch, Stmp4 * Sch);
    ast_builder->insert_assignment(Stmp1, Sfour_gamma_squared * Stmp1);
    ast_builder->insert_assignment(
        Stmp1, bit_cast<Tf>(expr_select(
                   Stmp2 <= Stmp1, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Ssine_pi_over_eight, Stmp1));
    ast_builder->insert_assignment(
        Ssh, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Ssh));
    ast_builder->insert_assignment(Ssh, svd_bitwise_or<Tf, Ti>(Ssh, Stmp2));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Scosine_pi_over_eight, Stmp1));
    ast_builder->insert_assignment(
        Sch, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Sch));
    ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp2));
    ast_builder->insert_assignment(Stmp1, Ssh * Ssh);
    ast_builder->insert_assignment(Stmp2, Sch * Sch);
    ast_builder->insert_assignment(Sc, Stmp2 - Stmp1);
    ast_builder->insert_assignment(Ss, Sch * Ssh);
    ast_builder->insert_assignment(Ss, Ss + Ss);
    ast_builder->insert_assignment(Stmp3, Stmp1 + Stmp2);
    ast_builder->insert_assignment(Ss33, Ss33 * Stmp3);
    ast_builder->insert_assignment(Ss31, Ss31 * Stmp3);
    ast_builder->insert_assignment(Ss32, Ss32 * Stmp3);
    ast_builder->insert_assignment(Ss33, Ss33 * Stmp3);
    ast_builder->insert_assignment(Stmp1, Ss * Ss31);
    ast_builder->insert_assignment(Stmp2, Ss * Ss32);
    ast_builder->insert_assignment(Ss31, Sc * Ss31);
    ast_builder->insert_assignment(Ss32, Sc * Ss32);
    ast_builder->insert_assignment(Ss31, Stmp2 + Ss31);
    ast_builder->insert_assignment(Ss32, Ss32 - Stmp1);
    ast_builder->insert_assignment(Stmp2, Ss * Ss);
    ast_builder->insert_assignment(Stmp1, Ss22 * Stmp2);
    ast_builder->insert_assignment(Stmp3, Ss11 * Stmp2);
    ast_builder->insert_assignment(Stmp4, Sc * Sc);
    ast_builder->insert_assignment(Ss11, Ss11 * Stmp4);
    ast_builder->insert_assignment(Ss22, Ss22 * Stmp4);
    ast_builder->insert_assignment(Ss11, Ss11 + Stmp1);
    ast_builder->insert_assignment(Ss22, Ss22 + Stmp3);
    ast_builder->insert_assignment(Stmp4, Stmp4 - Stmp2);
    ast_builder->insert_assignment(Stmp2, Ss21 + Ss21);
    ast_builder->insert_assignment(Ss21, Ss21 * Stmp4);
    ast_builder->insert_assignment(Stmp4, Sc * Ss);
    ast_builder->insert_assignment(Stmp2, Stmp2 * Stmp4);
    ast_builder->insert_assignment(Stmp5, Stmp5 * Stmp4);
    ast_builder->insert_assignment(Ss11, Ss11 + Stmp2);
    ast_builder->insert_assignment(Ss21, Ss21 - Stmp5);
    ast_builder->insert_assignment(Ss22, Ss22 - Stmp2);
    ast_builder->insert_assignment(Stmp1, Ssh * Sqvvx);
    ast_builder->insert_assignment(Stmp2, Ssh * Sqvvy);
    ast_builder->insert_assignment(Stmp3, Ssh * Sqvvz);
    ast_builder->insert_assignment(Ssh, Ssh * Sqvs);
    ast_builder->insert_assignment(Sqvs, Sch * Sqvs);
    ast_builder->insert_assignment(Sqvvx, Sch * Sqvvx);
    ast_builder->insert_assignment(Sqvvy, Sch * Sqvvy);
    ast_builder->insert_assignment(Sqvvz, Sch * Sqvvz);
    ast_builder->insert_assignment(Sqvvz, Sqvvz + Ssh);
    ast_builder->insert_assignment(Sqvs, Sqvs - Stmp3);
    ast_builder->insert_assignment(Sqvvx, Sqvvx + Stmp2);
    ast_builder->insert_assignment(Sqvvy, Sqvvy - Stmp1);
    ast_builder->insert_assignment(Ssh, Ss32 * Sone_half);
    ast_builder->insert_assignment(Stmp5, Ss22 - Ss33);
    ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
    ast_builder->insert_assignment(
        Stmp1,
        bit_cast<Tf>(expr_select(Stmp2 >= Stiny_number,
                                 Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
    ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Stmp1, Ssh));
    ast_builder->insert_assignment(Sch, svd_bitwise_and<Tf, Ti>(Stmp1, Stmp5));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Sone));
    ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp2));
    ast_builder->insert_assignment(Stmp1, Ssh * Ssh);
    ast_builder->insert_assignment(Stmp2, Sch * Sch);
    ast_builder->insert_assignment(Stmp3, Stmp1 + Stmp2);
    ast_builder->insert_assignment(Stmp4, rsqrt(Stmp3));
    ast_builder->insert_assignment(Ssh, Stmp4 * Ssh);
    ast_builder->insert_assignment(Sch, Stmp4 * Sch);
    ast_builder->insert_assignment(Stmp1, Sfour_gamma_squared * Stmp1);
    ast_builder->insert_assignment(
        Stmp1, bit_cast<Tf>(expr_select(
                   Stmp2 <= Stmp1, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Ssine_pi_over_eight, Stmp1));
    ast_builder->insert_assignment(
        Ssh, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Ssh));
    ast_builder->insert_assignment(Ssh, svd_bitwise_or<Tf, Ti>(Ssh, Stmp2));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Scosine_pi_over_eight, Stmp1));
    ast_builder->insert_assignment(
        Sch, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Sch));
    ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp2));
    ast_builder->insert_assignment(Stmp1, Ssh * Ssh);
    ast_builder->insert_assignment(Stmp2, Sch * Sch);
    ast_builder->insert_assignment(Sc, Stmp2 - Stmp1);
    ast_builder->insert_assignment(Ss, Sch * Ssh);
    ast_builder->insert_assignment(Ss, Ss + Ss);
    ast_builder->insert_assignment(Stmp3, Stmp1 + Stmp2);
    ast_builder->insert_assignment(Ss11, Ss11 * Stmp3);
    ast_builder->insert_assignment(Ss21, Ss21 * Stmp3);
    ast_builder->insert_assignment(Ss31, Ss31 * Stmp3);
    ast_builder->insert_assignment(Ss11, Ss11 * Stmp3);
    ast_builder->insert_assignment(Stmp1, Ss * Ss21);
    ast_builder->insert_assignment(Stmp2, Ss * Ss31);
    ast_builder->insert_assignment(Ss21, Sc * Ss21);
    ast_builder->insert_assignment(Ss31, Sc * Ss31);
    ast_builder->insert_assignment(Ss21, Stmp2 + Ss21);
    ast_builder->insert_assignment(Ss31, Ss31 - Stmp1);
    ast_builder->insert_assignment(Stmp2, Ss * Ss);
    ast_builder->insert_assignment(Stmp1, Ss33 * Stmp2);
    ast_builder->insert_assignment(Stmp3, Ss22 * Stmp2);
    ast_builder->insert_assignment(Stmp4, Sc * Sc);
    ast_builder->insert_assignment(Ss22, Ss22 * Stmp4);
    ast_builder->insert_assignment(Ss33, Ss33 * Stmp4);
    ast_builder->insert_assignment(Ss22, Ss22 + Stmp1);
    ast_builder->insert_assignment(Ss33, Ss33 + Stmp3);
    ast_builder->insert_assignment(Stmp4, Stmp4 - Stmp2);
    ast_builder->insert_assignment(Stmp2, Ss32 + Ss32);
    ast_builder->insert_assignment(Ss32, Ss32 * Stmp4);
    ast_builder->insert_assignment(Stmp4, Sc * Ss);
    ast_builder->insert_assignment(Stmp2, Stmp2 * Stmp4);
    ast_builder->insert_assignment(Stmp5, Stmp5 * Stmp4);
    ast_builder->insert_assignment(Ss22, Ss22 + Stmp2);
    ast_builder->insert_assignment(Ss32, Ss32 - Stmp5);
    ast_builder->insert_assignment(Ss33, Ss33 - Stmp2);
    ast_builder->insert_assignment(Stmp1, Ssh * Sqvvx);
    ast_builder->insert_assignment(Stmp2, Ssh * Sqvvy);
    ast_builder->insert_assignment(Stmp3, Ssh * Sqvvz);
    ast_builder->insert_assignment(Ssh, Ssh * Sqvs);
    ast_builder->insert_assignment(Sqvs, Sch * Sqvs);
    ast_builder->insert_assignment(Sqvvx, Sch * Sqvvx);
    ast_builder->insert_assignment(Sqvvy, Sch * Sqvvy);
    ast_builder->insert_assignment(Sqvvz, Sch * Sqvvz);
    ast_builder->insert_assignment(Sqvvx, Sqvvx + Ssh);
    ast_builder->insert_assignment(Sqvs, Sqvs - Stmp1);
    ast_builder->insert_assignment(Sqvvy, Sqvvy + Stmp3);
    ast_builder->insert_assignment(Sqvvz, Sqvvz - Stmp2);
    ast_builder->insert_assignment(Ssh, Ss31 * Sone_half);
    ast_builder->insert_assignment(Stmp5, Ss33 - Ss11);
    ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
    ast_builder->insert_assignment(
        Stmp1,
        bit_cast<Tf>(expr_select(Stmp2 >= Stiny_number,
                                 Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
    ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Stmp1, Ssh));
    ast_builder->insert_assignment(Sch, svd_bitwise_and<Tf, Ti>(Stmp1, Stmp5));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Sone));
    ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp2));
    ast_builder->insert_assignment(Stmp1, Ssh * Ssh);
    ast_builder->insert_assignment(Stmp2, Sch * Sch);
    ast_builder->insert_assignment(Stmp3, Stmp1 + Stmp2);
    ast_builder->insert_assignment(Stmp4, rsqrt(Stmp3));
    ast_builder->insert_assignment(Ssh, Stmp4 * Ssh);
    ast_builder->insert_assignment(Sch, Stmp4 * Sch);
    ast_builder->insert_assignment(Stmp1, Sfour_gamma_squared * Stmp1);
    ast_builder->insert_assignment(
        Stmp1, bit_cast<Tf>(expr_select(
                   Stmp2 <= Stmp1, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Ssine_pi_over_eight, Stmp1));
    ast_builder->insert_assignment(
        Ssh, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Ssh));
    ast_builder->insert_assignment(Ssh, svd_bitwise_or<Tf, Ti>(Ssh, Stmp2));
    ast_builder->insert_assignment(
        Stmp2, svd_bitwise_and<Tf, Ti>(Scosine_pi_over_eight, Stmp1));
    ast_builder->insert_assignment(
        Sch, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp1)), Sch));
    ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp2));
    ast_builder->insert_assignment(Stmp1, Ssh * Ssh);
    ast_builder->insert_assignment(Stmp2, Sch * Sch);
    ast_builder->insert_assignment(Sc, Stmp2 - Stmp1);
    ast_builder->insert_assignment(Ss, Sch * Ssh);
    ast_builder->insert_assignment(Ss, Ss + Ss);
    ast_builder->insert_assignment(Stmp3, Stmp1 + Stmp2);
    ast_builder->insert_assignment(Ss22, Ss22 * Stmp3);
    ast_builder->insert_assignment(Ss32, Ss32 * Stmp3);
    ast_builder->insert_assignment(Ss21, Ss21 * Stmp3);
    ast_builder->insert_assignment(Ss22, Ss22 * Stmp3);
    ast_builder->insert_assignment(Stmp1, Ss * Ss32);
    ast_builder->insert_assignment(Stmp2, Ss * Ss21);
    ast_builder->insert_assignment(Ss32, Sc * Ss32);
    ast_builder->insert_assignment(Ss21, Sc * Ss21);
    ast_builder->insert_assignment(Ss32, Stmp2 + Ss32);
    ast_builder->insert_assignment(Ss21, Ss21 - Stmp1);
    ast_builder->insert_assignment(Stmp2, Ss * Ss);
    ast_builder->insert_assignment(Stmp1, Ss11 * Stmp2);
    ast_builder->insert_assignment(Stmp3, Ss33 * Stmp2);
    ast_builder->insert_assignment(Stmp4, Sc * Sc);
    ast_builder->insert_assignment(Ss33, Ss33 * Stmp4);
    ast_builder->insert_assignment(Ss11, Ss11 * Stmp4);
    ast_builder->insert_assignment(Ss33, Ss33 + Stmp1);
    ast_builder->insert_assignment(Ss11, Ss11 + Stmp3);
    ast_builder->insert_assignment(Stmp4, Stmp4 - Stmp2);
    ast_builder->insert_assignment(Stmp2, Ss31 + Ss31);
    ast_builder->insert_assignment(Ss31, Ss31 * Stmp4);
    ast_builder->insert_assignment(Stmp4, Sc * Ss);
    ast_builder->insert_assignment(Stmp2, Stmp2 * Stmp4);
    ast_builder->insert_assignment(Stmp5, Stmp5 * Stmp4);
    ast_builder->insert_assignment(Ss33, Ss33 + Stmp2);
    ast_builder->insert_assignment(Ss31, Ss31 - Stmp5);
    ast_builder->insert_assignment(Ss11, Ss11 - Stmp2);
    ast_builder->insert_assignment(Stmp1, Ssh * Sqvvx);
    ast_builder->insert_assignment(Stmp2, Ssh * Sqvvy);
    ast_builder->insert_assignment(Stmp3, Ssh * Sqvvz);
    ast_builder->insert_assignment(Ssh, Ssh * Sqvs);
    ast_builder->insert_assignment(Sqvs, Sch * Sqvs);
    ast_builder->insert_assignment(Sqvvx, Sch * Sqvvx);
    ast_builder->insert_assignment(Sqvvy, Sch * Sqvvy);
    ast_builder->insert_assignment(Sqvvz, Sch * Sqvvz);
    ast_builder->insert_assignment(Sqvvy, Sqvvy + Ssh);
    ast_builder->insert_assignment(Sqvs, Sqvs - Stmp2);
    ast_builder->insert_assignment(Sqvvz, Sqvvz + Stmp1);
    ast_builder->insert_assignment(Sqvvx, Sqvvx - Stmp3);
  });
  ast_builder->insert_assignment(Stmp2, Sqvs * Sqvs);
  ast_builder->insert_assignment(Stmp1, Sqvvx * Sqvvx);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, Sqvvy * Sqvvy);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, Sqvvz * Sqvvz);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Sqvs, Sqvs * Stmp1);
  ast_builder->insert_assignment(Sqvvx, Sqvvx * Stmp1);
  ast_builder->insert_assignment(Sqvvy, Sqvvy * Stmp1);
  ast_builder->insert_assignment(Sqvvz, Sqvvz * Stmp1);
  ast_builder->insert_assignment(Stmp1, Sqvvx * Sqvvx);
  ast_builder->insert_assignment(Stmp2, Sqvvy * Sqvvy);
  ast_builder->insert_assignment(Stmp3, Sqvvz * Sqvvz);
  ast_builder->insert_assignment(Sv11, Sqvs * Sqvs);
  ast_builder->insert_assignment(Sv22, Sv11 - Stmp1);
  ast_builder->insert_assignment(Sv33, Sv22 - Stmp2);
  ast_builder->insert_assignment(Sv33, Sv33 + Stmp3);
  ast_builder->insert_assignment(Sv22, Sv22 + Stmp2);
  ast_builder->insert_assignment(Sv22, Sv22 - Stmp3);
  ast_builder->insert_assignment(Sv11, Sv11 + Stmp1);
  ast_builder->insert_assignment(Sv11, Sv11 - Stmp2);
  ast_builder->insert_assignment(Sv11, Sv11 - Stmp3);
  ast_builder->insert_assignment(Stmp1, Sqvvx + Sqvvx);
  ast_builder->insert_assignment(Stmp2, Sqvvy + Sqvvy);
  ast_builder->insert_assignment(Stmp3, Sqvvz + Sqvvz);
  ast_builder->insert_assignment(Sv32, Sqvs * Stmp1);
  ast_builder->insert_assignment(Sv13, Sqvs * Stmp2);
  ast_builder->insert_assignment(Sv21, Sqvs * Stmp3);
  ast_builder->insert_assignment(Stmp1, Sqvvy * Stmp1);
  ast_builder->insert_assignment(Stmp2, Sqvvz * Stmp2);
  ast_builder->insert_assignment(Stmp3, Sqvvx * Stmp3);
  ast_builder->insert_assignment(Sv12, Stmp1 - Sv21);
  ast_builder->insert_assignment(Sv23, Stmp2 - Sv32);
  ast_builder->insert_assignment(Sv31, Stmp3 - Sv13);
  ast_builder->insert_assignment(Sv21, Stmp1 + Sv21);
  ast_builder->insert_assignment(Sv32, Stmp2 + Sv32);
  ast_builder->insert_assignment(Sv13, Stmp3 + Sv13);
  ast_builder->insert_assignment(Stmp2, Sa12);
  ast_builder->insert_assignment(Stmp3, Sa13);
  ast_builder->insert_assignment(Sa12, Sv12 * Sa11);
  ast_builder->insert_assignment(Sa13, Sv13 * Sa11);
  ast_builder->insert_assignment(Sa11, Sv11 * Sa11);
  ast_builder->insert_assignment(Stmp1, Sv21 * Stmp2);
  ast_builder->insert_assignment(Sa11, Sa11 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv31 * Stmp3);
  ast_builder->insert_assignment(Sa11, Sa11 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv22 * Stmp2);
  ast_builder->insert_assignment(Sa12, Sa12 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv32 * Stmp3);
  ast_builder->insert_assignment(Sa12, Sa12 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv23 * Stmp2);
  ast_builder->insert_assignment(Sa13, Sa13 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv33 * Stmp3);
  ast_builder->insert_assignment(Sa13, Sa13 + Stmp1);
  ast_builder->insert_assignment(Stmp2, Sa22);
  ast_builder->insert_assignment(Stmp3, Sa23);
  ast_builder->insert_assignment(Sa22, Sv12 * Sa21);
  ast_builder->insert_assignment(Sa23, Sv13 * Sa21);
  ast_builder->insert_assignment(Sa21, Sv11 * Sa21);
  ast_builder->insert_assignment(Stmp1, Sv21 * Stmp2);
  ast_builder->insert_assignment(Sa21, Sa21 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv31 * Stmp3);
  ast_builder->insert_assignment(Sa21, Sa21 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv22 * Stmp2);
  ast_builder->insert_assignment(Sa22, Sa22 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv32 * Stmp3);
  ast_builder->insert_assignment(Sa22, Sa22 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv23 * Stmp2);
  ast_builder->insert_assignment(Sa23, Sa23 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv33 * Stmp3);
  ast_builder->insert_assignment(Sa23, Sa23 + Stmp1);
  ast_builder->insert_assignment(Stmp2, Sa32);
  ast_builder->insert_assignment(Stmp3, Sa33);
  ast_builder->insert_assignment(Sa32, Sv12 * Sa31);
  ast_builder->insert_assignment(Sa33, Sv13 * Sa31);
  ast_builder->insert_assignment(Sa31, Sv11 * Sa31);
  ast_builder->insert_assignment(Stmp1, Sv21 * Stmp2);
  ast_builder->insert_assignment(Sa31, Sa31 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv31 * Stmp3);
  ast_builder->insert_assignment(Sa31, Sa31 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv22 * Stmp2);
  ast_builder->insert_assignment(Sa32, Sa32 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv32 * Stmp3);
  ast_builder->insert_assignment(Sa32, Sa32 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv23 * Stmp2);
  ast_builder->insert_assignment(Sa33, Sa33 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sv33 * Stmp3);
  ast_builder->insert_assignment(Sa33, Sa33 + Stmp1);
  ast_builder->insert_assignment(Stmp1, Sa11 * Sa11);
  ast_builder->insert_assignment(Stmp4, Sa21 * Sa21);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp4, Sa31 * Sa31);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp2, Sa12 * Sa12);
  ast_builder->insert_assignment(Stmp4, Sa22 * Sa22);
  ast_builder->insert_assignment(Stmp2, Stmp2 + Stmp4);
  ast_builder->insert_assignment(Stmp4, Sa32 * Sa32);
  ast_builder->insert_assignment(Stmp2, Stmp2 + Stmp4);
  ast_builder->insert_assignment(Stmp3, Sa13 * Sa13);
  ast_builder->insert_assignment(Stmp4, Sa23 * Sa23);
  ast_builder->insert_assignment(Stmp3, Stmp3 + Stmp4);
  ast_builder->insert_assignment(Stmp4, Sa33 * Sa33);
  ast_builder->insert_assignment(Stmp3, Stmp3 + Stmp4);
  ast_builder->insert_assignment(
      Stmp4, bit_cast<Tf>(expr_select(
                 Stmp1 < Stmp2, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa11, Sa12));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa11, svd_bitwise_xor<Tf, Ti>(Sa11, Stmp5));
  ast_builder->insert_assignment(Sa12, svd_bitwise_xor<Tf, Ti>(Sa12, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa21, Sa22));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa21, svd_bitwise_xor<Tf, Ti>(Sa21, Stmp5));
  ast_builder->insert_assignment(Sa22, svd_bitwise_xor<Tf, Ti>(Sa22, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa31, Sa32));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa31, svd_bitwise_xor<Tf, Ti>(Sa31, Stmp5));
  ast_builder->insert_assignment(Sa32, svd_bitwise_xor<Tf, Ti>(Sa32, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv11, Sv12));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv11, svd_bitwise_xor<Tf, Ti>(Sv11, Stmp5));
  ast_builder->insert_assignment(Sv12, svd_bitwise_xor<Tf, Ti>(Sv12, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv21, Sv22));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv21, svd_bitwise_xor<Tf, Ti>(Sv21, Stmp5));
  ast_builder->insert_assignment(Sv22, svd_bitwise_xor<Tf, Ti>(Sv22, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv31, Sv32));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv31, svd_bitwise_xor<Tf, Ti>(Sv31, Stmp5));
  ast_builder->insert_assignment(Sv32, svd_bitwise_xor<Tf, Ti>(Sv32, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Stmp1, Stmp2));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Stmp1, svd_bitwise_xor<Tf, Ti>(Stmp1, Stmp5));
  ast_builder->insert_assignment(Stmp2, svd_bitwise_xor<Tf, Ti>(Stmp2, Stmp5));
  ast_builder->insert_assignment(Stmp5, Expr(Tf(-2.0f)));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Stmp4, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Stmp4, Stmp4 + Stmp5);
  ast_builder->insert_assignment(Sa12, Sa12 * Stmp4);
  ast_builder->insert_assignment(Sa22, Sa22 * Stmp4);
  ast_builder->insert_assignment(Sa32, Sa32 * Stmp4);
  ast_builder->insert_assignment(Sv12, Sv12 * Stmp4);
  ast_builder->insert_assignment(Sv22, Sv22 * Stmp4);
  ast_builder->insert_assignment(Sv32, Sv32 * Stmp4);
  ast_builder->insert_assignment(
      Stmp4, bit_cast<Tf>(expr_select(
                 Stmp1 < Stmp3, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa11, Sa13));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa11, svd_bitwise_xor<Tf, Ti>(Sa11, Stmp5));
  ast_builder->insert_assignment(Sa13, svd_bitwise_xor<Tf, Ti>(Sa13, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa21, Sa23));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa21, svd_bitwise_xor<Tf, Ti>(Sa21, Stmp5));
  ast_builder->insert_assignment(Sa23, svd_bitwise_xor<Tf, Ti>(Sa23, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa31, Sa33));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa31, svd_bitwise_xor<Tf, Ti>(Sa31, Stmp5));
  ast_builder->insert_assignment(Sa33, svd_bitwise_xor<Tf, Ti>(Sa33, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv11, Sv13));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv11, svd_bitwise_xor<Tf, Ti>(Sv11, Stmp5));
  ast_builder->insert_assignment(Sv13, svd_bitwise_xor<Tf, Ti>(Sv13, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv21, Sv23));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv21, svd_bitwise_xor<Tf, Ti>(Sv21, Stmp5));
  ast_builder->insert_assignment(Sv23, svd_bitwise_xor<Tf, Ti>(Sv23, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv31, Sv33));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv31, svd_bitwise_xor<Tf, Ti>(Sv31, Stmp5));
  ast_builder->insert_assignment(Sv33, svd_bitwise_xor<Tf, Ti>(Sv33, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Stmp1, Stmp3));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Stmp1, svd_bitwise_xor<Tf, Ti>(Stmp1, Stmp5));
  ast_builder->insert_assignment(Stmp3, svd_bitwise_xor<Tf, Ti>(Stmp3, Stmp5));
  ast_builder->insert_assignment(Stmp5, Expr(Tf(-2.0f)));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Stmp4, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Stmp4, Stmp4 + Stmp5);
  ast_builder->insert_assignment(Sa11, Sa11 * Stmp4);
  ast_builder->insert_assignment(Sa21, Sa21 * Stmp4);
  ast_builder->insert_assignment(Sa31, Sa31 * Stmp4);
  ast_builder->insert_assignment(Sv11, Sv11 * Stmp4);
  ast_builder->insert_assignment(Sv21, Sv21 * Stmp4);
  ast_builder->insert_assignment(Sv31, Sv31 * Stmp4);
  ast_builder->insert_assignment(
      Stmp4, bit_cast<Tf>(expr_select(
                 Stmp2 < Stmp3, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa12, Sa13));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa12, svd_bitwise_xor<Tf, Ti>(Sa12, Stmp5));
  ast_builder->insert_assignment(Sa13, svd_bitwise_xor<Tf, Ti>(Sa13, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa22, Sa23));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa22, svd_bitwise_xor<Tf, Ti>(Sa22, Stmp5));
  ast_builder->insert_assignment(Sa23, svd_bitwise_xor<Tf, Ti>(Sa23, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sa32, Sa33));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sa32, svd_bitwise_xor<Tf, Ti>(Sa32, Stmp5));
  ast_builder->insert_assignment(Sa33, svd_bitwise_xor<Tf, Ti>(Sa33, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv12, Sv13));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv12, svd_bitwise_xor<Tf, Ti>(Sv12, Stmp5));
  ast_builder->insert_assignment(Sv13, svd_bitwise_xor<Tf, Ti>(Sv13, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv22, Sv23));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv22, svd_bitwise_xor<Tf, Ti>(Sv22, Stmp5));
  ast_builder->insert_assignment(Sv23, svd_bitwise_xor<Tf, Ti>(Sv23, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Sv32, Sv33));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Sv32, svd_bitwise_xor<Tf, Ti>(Sv32, Stmp5));
  ast_builder->insert_assignment(Sv33, svd_bitwise_xor<Tf, Ti>(Sv33, Stmp5));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_xor<Tf, Ti>(Stmp2, Stmp3));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Stmp2, svd_bitwise_xor<Tf, Ti>(Stmp2, Stmp5));
  ast_builder->insert_assignment(Stmp3, svd_bitwise_xor<Tf, Ti>(Stmp3, Stmp5));
  ast_builder->insert_assignment(Stmp5, Expr(Tf(-2.0f)));
  ast_builder->insert_assignment(Stmp5, svd_bitwise_and<Tf, Ti>(Stmp5, Stmp4));
  ast_builder->insert_assignment(Stmp4, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Stmp4, Stmp4 + Stmp5);
  ast_builder->insert_assignment(Sa13, Sa13 * Stmp4);
  ast_builder->insert_assignment(Sa23, Sa23 * Stmp4);
  ast_builder->insert_assignment(Sa33, Sa33 * Stmp4);
  ast_builder->insert_assignment(Sv13, Sv13 * Stmp4);
  ast_builder->insert_assignment(Sv23, Sv23 * Stmp4);
  ast_builder->insert_assignment(Sv33, Sv33 * Stmp4);
  ast_builder->insert_assignment(Su11, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Su21, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Su31, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Su12, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Su22, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Su32, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Su13, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Su23, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Su33, Expr(Tf(1.0f)));
  ast_builder->insert_assignment(Ssh, Sa21 * Sa21);
  ast_builder->insert_assignment(
      Ssh, bit_cast<Tf>(expr_select(Ssh >= Ssmall_number,
                                    Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Ssh, Sa21));
  ast_builder->insert_assignment(Stmp5, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Sch, Stmp5 - Sa11);
  ast_builder->insert_assignment(Sch, max(Sch, Sa11));
  ast_builder->insert_assignment(Sch, max(Sch, Ssmall_number));
  ast_builder->insert_assignment(
      Stmp5, bit_cast<Tf>(expr_select(
                 Sa11 >= Stmp5, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Stmp1, Sch * Sch);
  ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 * Stmp2);
  ast_builder->insert_assignment(Sch, Sch + Stmp1);
  ast_builder->insert_assignment(
      Stmp1, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp5)), Ssh));
  ast_builder->insert_assignment(
      Stmp2, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp5)), Sch));
  ast_builder->insert_assignment(Sch, svd_bitwise_and<Tf, Ti>(Stmp5, Sch));
  ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Stmp5, Ssh));
  ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp1));
  ast_builder->insert_assignment(Ssh, svd_bitwise_or<Tf, Ti>(Ssh, Stmp2));
  ast_builder->insert_assignment(Stmp1, Sch * Sch);
  ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Sch, Sch * Stmp1);
  ast_builder->insert_assignment(Ssh, Ssh * Stmp1);
  ast_builder->insert_assignment(Sc, Sch * Sch);
  ast_builder->insert_assignment(Ss, Ssh * Ssh);
  ast_builder->insert_assignment(Sc, Sc - Ss);
  ast_builder->insert_assignment(Ss, Ssh * Sch);
  ast_builder->insert_assignment(Ss, Ss + Ss);
  ast_builder->insert_assignment(Stmp1, Ss * Sa11);
  ast_builder->insert_assignment(Stmp2, Ss * Sa21);
  ast_builder->insert_assignment(Sa11, Sc * Sa11);
  ast_builder->insert_assignment(Sa21, Sc * Sa21);
  ast_builder->insert_assignment(Sa11, Sa11 + Stmp2);
  ast_builder->insert_assignment(Sa21, Sa21 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Sa12);
  ast_builder->insert_assignment(Stmp2, Ss * Sa22);
  ast_builder->insert_assignment(Sa12, Sc * Sa12);
  ast_builder->insert_assignment(Sa22, Sc * Sa22);
  ast_builder->insert_assignment(Sa12, Sa12 + Stmp2);
  ast_builder->insert_assignment(Sa22, Sa22 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Sa13);
  ast_builder->insert_assignment(Stmp2, Ss * Sa23);
  ast_builder->insert_assignment(Sa13, Sc * Sa13);
  ast_builder->insert_assignment(Sa23, Sc * Sa23);
  ast_builder->insert_assignment(Sa13, Sa13 + Stmp2);
  ast_builder->insert_assignment(Sa23, Sa23 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su11);
  ast_builder->insert_assignment(Stmp2, Ss * Su12);
  ast_builder->insert_assignment(Su11, Sc * Su11);
  ast_builder->insert_assignment(Su12, Sc * Su12);
  ast_builder->insert_assignment(Su11, Su11 + Stmp2);
  ast_builder->insert_assignment(Su12, Su12 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su21);
  ast_builder->insert_assignment(Stmp2, Ss * Su22);
  ast_builder->insert_assignment(Su21, Sc * Su21);
  ast_builder->insert_assignment(Su22, Sc * Su22);
  ast_builder->insert_assignment(Su21, Su21 + Stmp2);
  ast_builder->insert_assignment(Su22, Su22 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su31);
  ast_builder->insert_assignment(Stmp2, Ss * Su32);
  ast_builder->insert_assignment(Su31, Sc * Su31);
  ast_builder->insert_assignment(Su32, Sc * Su32);
  ast_builder->insert_assignment(Su31, Su31 + Stmp2);
  ast_builder->insert_assignment(Su32, Su32 - Stmp1);
  ast_builder->insert_assignment(Ssh, Sa31 * Sa31);
  ast_builder->insert_assignment(
      Ssh, bit_cast<Tf>(expr_select(Ssh >= Ssmall_number,
                                    Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Ssh, Sa31));
  ast_builder->insert_assignment(Stmp5, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Sch, Stmp5 - Sa11);
  ast_builder->insert_assignment(Sch, max(Sch, Sa11));
  ast_builder->insert_assignment(Sch, max(Sch, Ssmall_number));
  ast_builder->insert_assignment(
      Stmp5, bit_cast<Tf>(expr_select(
                 Sa11 >= Stmp5, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Stmp1, Sch * Sch);
  ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 * Stmp2);
  ast_builder->insert_assignment(Sch, Sch + Stmp1);
  ast_builder->insert_assignment(
      Stmp1, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp5)), Ssh));
  ast_builder->insert_assignment(
      Stmp2, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp5)), Sch));
  ast_builder->insert_assignment(Sch, svd_bitwise_and<Tf, Ti>(Stmp5, Sch));
  ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Stmp5, Ssh));
  ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp1));
  ast_builder->insert_assignment(Ssh, svd_bitwise_or<Tf, Ti>(Ssh, Stmp2));
  ast_builder->insert_assignment(Stmp1, Sch * Sch);
  ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Sch, Sch * Stmp1);
  ast_builder->insert_assignment(Ssh, Ssh * Stmp1);
  ast_builder->insert_assignment(Sc, Sch * Sch);
  ast_builder->insert_assignment(Ss, Ssh * Ssh);
  ast_builder->insert_assignment(Sc, Sc - Ss);
  ast_builder->insert_assignment(Ss, Ssh * Sch);
  ast_builder->insert_assignment(Ss, Ss + Ss);
  ast_builder->insert_assignment(Stmp1, Ss * Sa11);
  ast_builder->insert_assignment(Stmp2, Ss * Sa31);
  ast_builder->insert_assignment(Sa11, Sc * Sa11);
  ast_builder->insert_assignment(Sa31, Sc * Sa31);
  ast_builder->insert_assignment(Sa11, Sa11 + Stmp2);
  ast_builder->insert_assignment(Sa31, Sa31 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Sa12);
  ast_builder->insert_assignment(Stmp2, Ss * Sa32);
  ast_builder->insert_assignment(Sa12, Sc * Sa12);
  ast_builder->insert_assignment(Sa32, Sc * Sa32);
  ast_builder->insert_assignment(Sa12, Sa12 + Stmp2);
  ast_builder->insert_assignment(Sa32, Sa32 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Sa13);
  ast_builder->insert_assignment(Stmp2, Ss * Sa33);
  ast_builder->insert_assignment(Sa13, Sc * Sa13);
  ast_builder->insert_assignment(Sa33, Sc * Sa33);
  ast_builder->insert_assignment(Sa13, Sa13 + Stmp2);
  ast_builder->insert_assignment(Sa33, Sa33 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su11);
  ast_builder->insert_assignment(Stmp2, Ss * Su13);
  ast_builder->insert_assignment(Su11, Sc * Su11);
  ast_builder->insert_assignment(Su13, Sc * Su13);
  ast_builder->insert_assignment(Su11, Su11 + Stmp2);
  ast_builder->insert_assignment(Su13, Su13 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su21);
  ast_builder->insert_assignment(Stmp2, Ss * Su23);
  ast_builder->insert_assignment(Su21, Sc * Su21);
  ast_builder->insert_assignment(Su23, Sc * Su23);
  ast_builder->insert_assignment(Su21, Su21 + Stmp2);
  ast_builder->insert_assignment(Su23, Su23 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su31);
  ast_builder->insert_assignment(Stmp2, Ss * Su33);
  ast_builder->insert_assignment(Su31, Sc * Su31);
  ast_builder->insert_assignment(Su33, Sc * Su33);
  ast_builder->insert_assignment(Su31, Su31 + Stmp2);
  ast_builder->insert_assignment(Su33, Su33 - Stmp1);
  ast_builder->insert_assignment(Ssh, Sa32 * Sa32);
  ast_builder->insert_assignment(
      Ssh, bit_cast<Tf>(expr_select(Ssh >= Ssmall_number,
                                    Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Ssh, Sa32));
  ast_builder->insert_assignment(Stmp5, Expr(Tf(0.0f)));
  ast_builder->insert_assignment(Sch, Stmp5 - Sa22);
  ast_builder->insert_assignment(Sch, max(Sch, Sa22));
  ast_builder->insert_assignment(Sch, max(Sch, Ssmall_number));
  ast_builder->insert_assignment(
      Stmp5, bit_cast<Tf>(expr_select(
                 Sa22 >= Stmp5, Expr(Ti(int32(0xffffffff))), Expr(Ti(0)))));
  ast_builder->insert_assignment(Stmp1, Sch * Sch);
  ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 * Stmp2);
  ast_builder->insert_assignment(Sch, Sch + Stmp1);
  ast_builder->insert_assignment(
      Stmp1, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp5)), Ssh));
  ast_builder->insert_assignment(
      Stmp2, svd_bitwise_and<Tf, Ti>(Expr(~bit_cast<Ti>(Stmp5)), Sch));
  ast_builder->insert_assignment(Sch, svd_bitwise_and<Tf, Ti>(Stmp5, Sch));
  ast_builder->insert_assignment(Ssh, svd_bitwise_and<Tf, Ti>(Stmp5, Ssh));
  ast_builder->insert_assignment(Sch, svd_bitwise_or<Tf, Ti>(Sch, Stmp1));
  ast_builder->insert_assignment(Ssh, svd_bitwise_or<Tf, Ti>(Ssh, Stmp2));
  ast_builder->insert_assignment(Stmp1, Sch * Sch);
  ast_builder->insert_assignment(Stmp2, Ssh * Ssh);
  ast_builder->insert_assignment(Stmp2, Stmp1 + Stmp2);
  ast_builder->insert_assignment(Stmp1, rsqrt(Stmp2));
  ast_builder->insert_assignment(Stmp4, Stmp1 * Sone_half);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp4);
  ast_builder->insert_assignment(Stmp3, Stmp1 * Stmp3);
  ast_builder->insert_assignment(Stmp3, Stmp2 * Stmp3);
  ast_builder->insert_assignment(Stmp1, Stmp1 + Stmp4);
  ast_builder->insert_assignment(Stmp1, Stmp1 - Stmp3);
  ast_builder->insert_assignment(Sch, Sch * Stmp1);
  ast_builder->insert_assignment(Ssh, Ssh * Stmp1);
  ast_builder->insert_assignment(Sc, Sch * Sch);
  ast_builder->insert_assignment(Ss, Ssh * Ssh);
  ast_builder->insert_assignment(Sc, Sc - Ss);
  ast_builder->insert_assignment(Ss, Ssh * Sch);
  ast_builder->insert_assignment(Ss, Ss + Ss);
  ast_builder->insert_assignment(Stmp1, Ss * Sa21);
  ast_builder->insert_assignment(Stmp2, Ss * Sa31);
  ast_builder->insert_assignment(Sa21, Sc * Sa21);
  ast_builder->insert_assignment(Sa31, Sc * Sa31);
  ast_builder->insert_assignment(Sa21, Sa21 + Stmp2);
  ast_builder->insert_assignment(Sa31, Sa31 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Sa22);
  ast_builder->insert_assignment(Stmp2, Ss * Sa32);
  ast_builder->insert_assignment(Sa22, Sc * Sa22);
  ast_builder->insert_assignment(Sa32, Sc * Sa32);
  ast_builder->insert_assignment(Sa22, Sa22 + Stmp2);
  ast_builder->insert_assignment(Sa32, Sa32 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Sa23);
  ast_builder->insert_assignment(Stmp2, Ss * Sa33);
  ast_builder->insert_assignment(Sa23, Sc * Sa23);
  ast_builder->insert_assignment(Sa33, Sc * Sa33);
  ast_builder->insert_assignment(Sa23, Sa23 + Stmp2);
  ast_builder->insert_assignment(Sa33, Sa33 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su12);
  ast_builder->insert_assignment(Stmp2, Ss * Su13);
  ast_builder->insert_assignment(Su12, Sc * Su12);
  ast_builder->insert_assignment(Su13, Sc * Su13);
  ast_builder->insert_assignment(Su12, Su12 + Stmp2);
  ast_builder->insert_assignment(Su13, Su13 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su22);
  ast_builder->insert_assignment(Stmp2, Ss * Su23);
  ast_builder->insert_assignment(Su22, Sc * Su22);
  ast_builder->insert_assignment(Su23, Sc * Su23);
  ast_builder->insert_assignment(Su22, Su22 + Stmp2);
  ast_builder->insert_assignment(Su23, Su23 - Stmp1);
  ast_builder->insert_assignment(Stmp1, Ss * Su32);
  ast_builder->insert_assignment(Stmp2, Ss * Su33);
  ast_builder->insert_assignment(Su32, Sc * Su32);
  ast_builder->insert_assignment(Su33, Sc * Su33);
  ast_builder->insert_assignment(Su32, Su32 + Stmp2);
  ast_builder->insert_assignment(Su33, Su33 - Stmp1);
  return std::make_tuple(Su11, Su12, Su13, Su21, Su22, Su23, Su31, Su32, Su33,
                         Sv11, Sv12, Sv13, Sv21, Sv22, Sv23, Sv31, Sv32, Sv33,
                         Sa11, Sa22, Sa33);
}

TLANG_NAMESPACE_END
