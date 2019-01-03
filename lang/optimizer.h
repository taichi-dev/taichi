#include "program.h"

TLANG_NAMESPACE_BEGIN

class Optimizer {
 public:
  virtual void run(Kernel &kernel) {
    while (search_and_replace(kernel.ret))
      ;
  }

  bool search_and_replace(Expr &expr) {
    for (auto &ch : expr->ch) {
      bool ret = search_and_replace(ch);
      if (ret)
        return true;
    }

    if (expr->type == NodeType::load || expr->type == NodeType::store) {
      auto &ptr = expr._pointer();
      auto &addr_node = expr._pointer()._address();
      bool all_same = true;
      for (int i = 0; i < addr_node->lanes; i++) {
        if (addr_node->new_addresses(i) != addr_node->new_addresses(0))
          all_same = false;
      }

      bool incremental = true;

      // only consider the last one.
      auto &index_node = expr._pointer()->ch.back();

      // TODO: check non-last indices are uniform
      if (index_node->type == NodeType::index) {
        int offset_start = index_node->index_offset(0);
        int offset_inc = index_node->index_offset(1) - offset_start;
        for (int i = 0; i + 1 < addr_node->lanes; i++) {
          if (index_node->index_offset(i) + offset_inc !=
              index_node->index_offset(i + 1)) {
            incremental = false;
          }
        }

        auto snode = addr_node->new_addresses(0);
        if (all_same && incremental && offset_start == 0 && offset_inc == 1) {
          if (snode->parent->type == SNodeType::fixed) {
            // replace load with vload
            if (expr->type == NodeType::load) {
              TC_INFO("Optimized load");
              auto vload = Expr::create(NodeType::vload, addr_node);
              vload->ch.resize(ptr->ch.size());
              for (int i = 1; i < ptr->ch.size(); i++) {
                vload->ch[i] = ptr->ch[i];
              }
              vload->set_similar(expr);
              vload->is_vectorized = true;
              expr = vload;
              return true;
            } else {
              TC_INFO("Optimized store");
              auto vstore =
                  Expr::create(NodeType::vstore, addr_node, expr->ch[1]);
              vstore->ch.resize(ptr->ch.size() + 1);
              for (int i = 1; i < ptr->ch.size(); i++) {
                vstore->ch[i + 1] = ptr->ch[i];
              }
              vstore->set_similar(expr);
              vstore->is_vectorized = true;
              expr = vstore;
              return true;
            }
          }
        }
      }
    }

    return false;
  }
};

TLANG_NAMESPACE_END
