#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class ConvertIntoLoopIndexStmt : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  static void convert(Stmt *loop, Stmt *loop_var, int index) {
    if (!loop_var)
      return;
    irpass::replace_statements_with(
        loop,
        [&](Stmt *load) {
          if (auto local_load = load->cast<LocalLoadStmt>()) {
            return local_load->width() == 1 &&
                   local_load->ptr[0].var == loop_var &&
                   local_load->ptr[0].offset == 0;
          }
          return false;
        },
        [&]() { return Stmt::make<LoopIndexStmt>(loop, index); });
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (auto range_for = stmt->cast<RangeForStmt>()) {
      convert(range_for, range_for->loop_var, 0);
      range_for->loop_var = nullptr;
    } else if (auto struct_for = stmt->cast<StructForStmt>()) {
      auto leaf = struct_for->snode;
      for (int i = 0; i < (int)struct_for->loop_vars.size(); i++) {
        convert(struct_for, struct_for->loop_vars[i],
                leaf->physical_index_position[i]);
        struct_for->loop_vars[i] = nullptr;
      }
    }
  }

  static void run(IRNode *node) {
    ConvertIntoLoopIndexStmt converter;
    while (true) {
      bool modified = false;
      try {
        node->accept(&converter);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {
void convert_into_loop_index(IRNode *root) {
  ConvertIntoLoopIndexStmt::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
