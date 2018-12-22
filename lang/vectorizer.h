#include "expr.h"
#include "visitor.h"

namespace taichi::Tlang {

class Vectorizer : public Visitor {
 public:
  std::map<Expr, Expr> scalar_to_vector;
  int simd_width;
  int group_size;
  int num_groups;

  Vectorizer(int simd_width)
      : Visitor(Visitor::Order::parent_first), simd_width(simd_width) {
  }

  void sort(Expr &expr);

  Expr run(Expr &expr, int group_size);

  void visit(Expr &expr) override;
};
}
