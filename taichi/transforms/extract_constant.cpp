#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class ExtractConstant : public BasicStmtVisitor {
 private:
  Block *top_level;

 public:
  using BasicStmtVisitor::visit;

  explicit ExtractConstant(IRNode *node) : top_level(nullptr) {
    if (node->is<Block>())
      top_level = node->as<Block>();
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(top_level);
    if (stmt->parent != top_level) {
      auto extracted = stmt->parent->extract(stmt);
      top_level->insert(std::move(extracted), 0);
      throw IRModified();
    }
  }

  void visit(OffloadedStmt *offload) override {
    if (offload->body) {
      Block *backup = top_level;
      top_level = offload->body.get();
      offload->body->accept(this);
      top_level = backup;
    }
  }

  static void run(IRNode *node) {
    ExtractConstant extractor(node);
    while (true) {
      bool modified = false;
      try {
        node->accept(&extractor);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {
void extract_constant(IRNode *root) {
  if (advanced_optimization)
    ExtractConstant::run(root);
}
}  // namespace irpass

TLANG_NAMESPACE_END
