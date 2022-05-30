// Arithmatic operations

#if defined(TI_EXPRESSION_IMPLEMENTATION)

#undef DEFINE_EXPRESSION_OP_UNARY
#undef DEFINE_EXPRESSION_FUNC_UNARY
#undef DEFINE_EXPRESSION_OP_BINARY
#undef DEFINE_EXPRESSION_FUNC_BINARY
#undef DEFINE_EXPRESSION_FUNC_TERNARY

#define DEFINE_EXPRESSION_OP_UNARY(op, opname)                       \
  Expr expr_##opname(const Expr &expr) {                             \
    return Expr::make<UnaryOpExpression>(UnaryOpType::opname, expr); \
  }                                                                  \
  Expr operator op(const Expr &expr) {                               \
    return expr_##opname(expr);                                      \
  }

#define DEFINE_EXPRESSION_FUNC_UNARY(opname)                         \
  Expr opname(const Expr &expr) {                                    \
    return Expr::make<UnaryOpExpression>(UnaryOpType::opname, expr); \
  }                                                                  \
  Expr expr_##opname(const Expr &expr) {                             \
    return opname(expr);                                             \
  }

#define DEFINE_EXPRESSION_OP_BINARY(op, opname)                            \
  Expr operator op(const Expr &lhs, const Expr &rhs) {                     \
    return Expr::make<BinaryOpExpression>(BinaryOpType::opname, lhs, rhs); \
  }                                                                        \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs) {                   \
    return lhs op rhs;                                                     \
  }

#define DEFINE_EXPRESSION_FUNC_BINARY(opname)                              \
  Expr opname(const Expr &lhs, const Expr &rhs) {                          \
    return Expr::make<BinaryOpExpression>(BinaryOpType::opname, lhs, rhs); \
  }                                                                        \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs) {                   \
    return opname(lhs, rhs);                                               \
  }

#define DEFINE_EXPRESSION_FUNC_TERNARY(opname)                               \
  Expr expr_##opname(const Expr &cond, const Expr &lhs, const Expr &rhs) {   \
    return Expr::make<TernaryOpExpression>(TernaryOpType::opname, cond, lhs, \
                                           rhs);                             \
  }                                                                          \
  Expr opname(const Expr &cond, const Expr &lhs, const Expr &rhs) {          \
    return expr_##opname(cond, lhs, rhs);                                    \
  }

#else

#define DEFINE_EXPRESSION_OP_UNARY(op, opname) \
  Expr operator op(const Expr &expr);          \
  Expr expr_##opname(const Expr &expr);

#define DEFINE_EXPRESSION_FUNC_UNARY(opname) \
  Expr opname(const Expr &expr);             \
  Expr expr_##opname(const Expr &expr);

#define DEFINE_EXPRESSION_OP_BINARY(op, opname)       \
  Expr operator op(const Expr &lhs, const Expr &rhs); \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs);

#define DEFINE_EXPRESSION_FUNC_BINARY(opname)    \
  Expr opname(const Expr &lhs, const Expr &rhs); \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs);

#define DEFINE_EXPRESSION_FUNC_TERNARY(opname)                     \
  Expr opname(const Expr &cond, const Expr &lhs, const Expr &rhs); \
  Expr expr_##opname(const Expr &cond, const Expr &lhs, const Expr &rhs);

#endif

namespace taichi {
namespace lang {

DEFINE_EXPRESSION_FUNC_UNARY(sqrt)
DEFINE_EXPRESSION_FUNC_UNARY(round)
DEFINE_EXPRESSION_FUNC_UNARY(floor)
DEFINE_EXPRESSION_FUNC_UNARY(ceil)
DEFINE_EXPRESSION_FUNC_UNARY(abs)
DEFINE_EXPRESSION_FUNC_UNARY(sin)
DEFINE_EXPRESSION_FUNC_UNARY(asin)
DEFINE_EXPRESSION_FUNC_UNARY(cos)
DEFINE_EXPRESSION_FUNC_UNARY(acos)
DEFINE_EXPRESSION_FUNC_UNARY(tan)
DEFINE_EXPRESSION_FUNC_UNARY(tanh)
DEFINE_EXPRESSION_FUNC_UNARY(inv)
DEFINE_EXPRESSION_FUNC_UNARY(rcp)
DEFINE_EXPRESSION_FUNC_UNARY(rsqrt)
DEFINE_EXPRESSION_FUNC_UNARY(exp)
DEFINE_EXPRESSION_FUNC_UNARY(log)
DEFINE_EXPRESSION_FUNC_UNARY(logic_not)
DEFINE_EXPRESSION_OP_UNARY(~, bit_not)
DEFINE_EXPRESSION_OP_UNARY(-, neg)

DEFINE_EXPRESSION_OP_BINARY(+, add)
DEFINE_EXPRESSION_OP_BINARY(-, sub)
DEFINE_EXPRESSION_OP_BINARY(*, mul)
DEFINE_EXPRESSION_OP_BINARY(/, div)
DEFINE_EXPRESSION_OP_BINARY(%, mod)
DEFINE_EXPRESSION_OP_BINARY(&&, logical_and)
DEFINE_EXPRESSION_OP_BINARY(||, logical_or)
DEFINE_EXPRESSION_OP_BINARY(&, bit_and)
DEFINE_EXPRESSION_OP_BINARY(|, bit_or)
DEFINE_EXPRESSION_OP_BINARY(^, bit_xor)
DEFINE_EXPRESSION_OP_BINARY(<<, bit_shl)
DEFINE_EXPRESSION_OP_BINARY(>>, bit_sar)
DEFINE_EXPRESSION_OP_BINARY(<, cmp_lt)
DEFINE_EXPRESSION_OP_BINARY(<=, cmp_le)
DEFINE_EXPRESSION_OP_BINARY(>, cmp_gt)
DEFINE_EXPRESSION_OP_BINARY(>=, cmp_ge)
DEFINE_EXPRESSION_OP_BINARY(==, cmp_eq)
DEFINE_EXPRESSION_OP_BINARY(!=, cmp_ne)

DEFINE_EXPRESSION_FUNC_BINARY(min)
DEFINE_EXPRESSION_FUNC_BINARY(max)
DEFINE_EXPRESSION_FUNC_BINARY(atan2)
DEFINE_EXPRESSION_FUNC_BINARY(pow)
DEFINE_EXPRESSION_FUNC_BINARY(truediv)
DEFINE_EXPRESSION_FUNC_BINARY(floordiv)
DEFINE_EXPRESSION_FUNC_BINARY(bit_shr)

DEFINE_EXPRESSION_FUNC_TERNARY(select)
DEFINE_EXPRESSION_FUNC_TERNARY(ifte)

}  // namespace lang
}  // namespace taichi

#undef DEFINE_EXPRESSION_OP_UNARY
#undef DEFINE_EXPRESSION_OP_BINARY
#undef DEFINE_EXPRESSION_FUNC_UNARY
#undef DEFINE_EXPRESSION_FUNC_BINARY
#undef DEFINE_EXPRESSION_FUNC_TERNARY
