#if defined(TC_EXPRESSION_IMPLEMENTATION)

#undef DEFINE_EXPRESSION_OP
#undef DEFINE_EXPRESSION_OP_UNARY
#undef DEFINE_EXPRESSION_FUNC

#define DEFINE_EXPRESSION_OP(op, op_name)                                 \
  Expr operator op(const Expr &lhs, const Expr &rhs) {                    \
    return Expr::make<BinaryOpExpression>(BinaryType::op_name, lhs, rhs); \
  }

#define DEFINE_EXPRESSION_OP_UNARY(opname)                         \
  Expr opname(Expr expr) {                                         \
    return Expr::make<UnaryOpExpression>(UnaryType::opname, expr); \
  }

#define DEFINE_EXPRESSION_FUNC(op_name)                                   \
  Expr op_name(const Expr &lhs, const Expr &rhs) {                        \
    return Expr::make<BinaryOpExpression>(BinaryType::op_name, lhs, rhs); \
  }

#else

#define DEFINE_EXPRESSION_OP(op, op_name) \
  Expr operator op(const Expr &lhs, const Expr &rhs);

#define DEFINE_EXPRESSION_OP_UNARY(opname) Expr opname(Expr expr);

#define DEFINE_EXPRESSION_FUNC(op_name) \
  Expr op_name(const Expr &lhs, const Expr &rhs);

#endif

DEFINE_EXPRESSION_OP_UNARY(sqrt)
DEFINE_EXPRESSION_OP_UNARY(floor)
DEFINE_EXPRESSION_OP_UNARY(abs)
DEFINE_EXPRESSION_OP_UNARY(sin)
DEFINE_EXPRESSION_OP_UNARY(cos)
DEFINE_EXPRESSION_OP_UNARY(inv)
DEFINE_EXPRESSION_OP_UNARY(rcp)
DEFINE_EXPRESSION_OP_UNARY(rsqrt)

DEFINE_EXPRESSION_OP(+, add)
DEFINE_EXPRESSION_OP(-, sub)
DEFINE_EXPRESSION_OP(*, mul)
DEFINE_EXPRESSION_OP(/, div)
DEFINE_EXPRESSION_OP(%, mod)
DEFINE_EXPRESSION_OP(&&, bit_and)
DEFINE_EXPRESSION_OP(||, bit_or)
DEFINE_EXPRESSION_OP(&, bit_and)
DEFINE_EXPRESSION_OP(|, bit_or)
DEFINE_EXPRESSION_OP (^, bit_xor)
DEFINE_EXPRESSION_OP(<, cmp_lt)
DEFINE_EXPRESSION_OP(<=, cmp_le)
DEFINE_EXPRESSION_OP(>, cmp_gt)
DEFINE_EXPRESSION_OP(>=, cmp_ge)
DEFINE_EXPRESSION_OP(==, cmp_eq)
DEFINE_EXPRESSION_OP(!=, cmp_ne)

DEFINE_EXPRESSION_FUNC(min);
DEFINE_EXPRESSION_FUNC(max);
