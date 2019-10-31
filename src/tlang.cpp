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
  std::function<void(SNode * node)> visit = [&](SNode *node) {
    std::vector<SNode *> places;
    for (auto &ch : node->ch) {
      if (ch->type == SNodeType::place) {
        if (!ch->is_primal())
          places.push_back(ch.get());
      } else {
        visit(ch.get());
      }
    }
    auto kernel_name = fmt::format("clear_gradient_{}", node->id);
    if (!places.empty()) {
      auto &ker = kernel([&] {
        if (places[0]->num_active_indices == 1) {
          For(*places[0]->expr, [&](Expr i) {
            for (auto s : places) {
              (*s->expr)[i] = 0;
            }
          });
        } else if (places[0]->num_active_indices == 2) {
          For(*places[0]->expr, [&](Expr i, Expr j) {
            for (auto s : places) {
              (*s->expr)[i, j] = 0;
            }
          });
        } else if (places[0]->num_active_indices == 3) {
          For(*places[0]->expr, [&](Expr i, Expr j, Expr k) {
            for (auto s : places) {
              (*s->expr)[i, j, k] = 0;
            }
          });
        } else if (places[0]->num_active_indices == 4) {
          For(*places[0]->expr, [&](Expr i, Expr j, Expr k, Expr l) {
            for (auto s : places) {
              (*s->expr)[i, j, k, l] = 0;
            }
          });
        } else if (places[0]->num_active_indices == 0) {
          for (auto s : places) {
            (*s->expr)[Expr(0)] = 0;
          }
        } else {
          TC_NOT_IMPLEMENTED
        }
      });
      ker.name = kernel_name;
      gradient_clearers.emplace_back([&] { ker(); });
    }
  };
  visit(&root);
}

void Program::get_snode_writer(SNode *snode) {
  TC_ASSERT(snode->type == SNodeType::place);
  auto kernel_name = fmt::format("snode_writer_{}", snode->id);
  auto &ker = kernel([&] {
    if (snode->num_active_indices == 1) {
      (*snode->expr)[Expr::make<ArgLoadExpression>(0)] =
          Expr::make<ArgLoadExpression>(1);
    } else {
      TC_NOT_IMPLEMENTED;
    }
  });
  ker.name = kernel_name;
}

TLANG_NAMESPACE_END
