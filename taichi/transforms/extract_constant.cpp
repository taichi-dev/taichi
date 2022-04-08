#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/compile_config.h"

TLANG_NAMESPACE_BEGIN

class ExtractConstant : public BasicStmtVisitor {
 private:
  Block *top_level_;
  DelayedIRModifier modifier_;

 public:
  using BasicStmtVisitor::visit;

  explicit ExtractConstant(IRNode *node) : top_level_(nullptr) {
    if (node->is<Block>())
      top_level_ = node->as<Block>();
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(top_level_);
    if (stmt->parent != top_level_) {
      modifier_.extract_to_block_front(stmt, top_level_);
    }
  }

  void visit(OffloadedStmt *offload) override {
    if (offload->body) {
      Block *backup = top_level_;
      top_level_ = offload->body.get();
      offload->body->accept(this);
      top_level_ = backup;
    }
  }

  static bool run(IRNode *node) {
    ExtractConstant extractor(node);
    bool ir_modified = false;
    while (true) {
      node->accept(&extractor);
      if (extractor.modifier_.modify_ir()) {
        ir_modified = true;
      } else {
        break;
      }
    }
    return ir_modified;
  }
};

namespace irpass {
bool extract_constant(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  if (config.advanced_optimization) {
    return ExtractConstant::run(root);
  } else {
    return false;
  }
}
}  // namespace irpass

TLANG_NAMESPACE_END
