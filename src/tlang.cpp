// Frontend constructs

#include "tlang.h"
#include "program.h"

TLANG_NAMESPACE_BEGIN

void layout(const std::function<void()> &body) {
  get_current_program().layout(body);
}

Expr global_new(Expr id_expr, DataType dt) {
  TC_ASSERT(id_expr.is<IdExpression>());
  auto ret = Expr(std::make_shared<GlobalVariableExpression>(
      dt, id_expr.cast<IdExpression>()->id));
  return ret;
}

Expr global_new(DataType dt, std::string name) {
  auto id_expr = std::make_shared<IdExpression>(name);
  return Expr::make<GlobalVariableExpression>(dt, id_expr->id);
}


void Program::initialize_gradient_clearers() {
  std::function<void(SNode *node)> visit = [&](SNode *node) {
    std::vector<SNode *> places;
    for (auto &ch: node->ch){
      if (ch->type == SNodeType::place) {
        places.push_back(ch.get());
      } else {
        visit(ch.get());
      }
    }
    auto kernel_name = fmt::format("clear_gradient_{}", node->id);
    if (!places.empty()) {
      TC_TAG;
      auto &ker = kernel([&] {
        if (places[0]->num_active_indices == 1) {
          For(*places[0]->expr, [&](Expr i){
            for (auto s: places) {
              (*s->expr)[i] = 0;
            }
          });
        }
      });
      ker.name = kernel_name;
      gradient_clearers.emplace_back([&]{
        TC_TAG;
        ker();
      });
    }
  };
  visit(&root);
}


TLANG_NAMESPACE_END
