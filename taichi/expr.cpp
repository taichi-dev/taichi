#include "expr.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

std::string Expr::serialize() const {
  TC_ASSERT(expr);
  return expr->serialize();
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


TLANG_NAMESPACE_END
