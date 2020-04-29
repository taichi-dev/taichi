// Arithmatic operations

#if defined(TI_EXPRESSION_IMPLEMENTATION)

#undef DEFINE_EXPRESSION_OP_BINARY
#undef DEFINE_EXPRESSION_OP_UNARY
#undef DEFINE_EXPRESSION_FUNC

#define DEFINE_EXPRESSION_OP_UNARY(opname)                           \
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

#define DEFINE_EXPRESSION_FUNC(opname)                                     \
  Expr opname(const Expr &lhs, const Expr &rhs) {                          \
    return Expr::make<BinaryOpExpression>(BinaryOpType::opname, lhs, rhs); \
  }                                                                        \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs) {                   \
    return opname(lhs, rhs);                                               \
  }

#else

#define DEFINE_EXPRESSION_OP_BINARY(op, opname)       \
  Expr operator op(const Expr &lhs, const Expr &rhs); \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs);

#define DEFINE_EXPRESSION_OP_UNARY(opname) \
  Expr opname(const Expr &expr);           \
  Expr expr_##opname(const Expr &expr);

#define DEFINE_EXPRESSION_FUNC(opname)           \
  Expr opname(const Expr &lhs, const Expr &rhs); \
  Expr expr_##opname(const Expr &lhs, const Expr &rhs);

#endif

DEFINE_EXPRESSION_OP_UNARY(sqrt)
DEFINE_EXPRESSION_OP_UNARY(floor)
DEFINE_EXPRESSION_OP_UNARY(ceil)
DEFINE_EXPRESSION_OP_UNARY(abs)
DEFINE_EXPRESSION_OP_UNARY(sin)
DEFINE_EXPRESSION_OP_UNARY(asin)
DEFINE_EXPRESSION_OP_UNARY(cos)
DEFINE_EXPRESSION_OP_UNARY(acos)
DEFINE_EXPRESSION_OP_UNARY(tan)
DEFINE_EXPRESSION_OP_UNARY(tanh)
DEFINE_EXPRESSION_OP_UNARY(inv)
DEFINE_EXPRESSION_OP_UNARY(rcp)
DEFINE_EXPRESSION_OP_UNARY(rsqrt)
DEFINE_EXPRESSION_OP_UNARY(exp)
DEFINE_EXPRESSION_OP_UNARY(log)

DEFINE_EXPRESSION_OP_BINARY(+, add)
DEFINE_EXPRESSION_OP_BINARY(-, sub)
DEFINE_EXPRESSION_OP_BINARY(*, mul)
DEFINE_EXPRESSION_OP_BINARY(/, div)
DEFINE_EXPRESSION_OP_BINARY(%, mod)
DEFINE_EXPRESSION_OP_BINARY(&&, bit_and)
DEFINE_EXPRESSION_OP_BINARY(||, bit_or)
// DEFINE_EXPRESSION_OP_BINARY(&, bit_and)
// DEFINE_EXPRESSION_OP_BINARY(|, bit_or)
DEFINE_EXPRESSION_OP_BINARY (^, bit_xor)
DEFINE_EXPRESSION_OP_BINARY(<, cmp_lt)
DEFINE_EXPRESSION_OP_BINARY(<=, cmp_le)
DEFINE_EXPRESSION_OP_BINARY(>, cmp_gt)
DEFINE_EXPRESSION_OP_BINARY(>=, cmp_ge)
DEFINE_EXPRESSION_OP_BINARY(==, cmp_eq)
DEFINE_EXPRESSION_OP_BINARY(!=, cmp_ne)

DEFINE_EXPRESSION_FUNC(min);
DEFINE_EXPRESSION_FUNC(max);
DEFINE_EXPRESSION_FUNC(atan2);
DEFINE_EXPRESSION_FUNC(pow);
DEFINE_EXPRESSION_FUNC(truediv);
DEFINE_EXPRESSION_FUNC(floordiv);

#undef DEFINE_EXPRESSION_OP_UNARY
#undef DEFINE_EXPRESSION_OP_BINARY
#undef DEFINE_EXPRESSION_FUNC
