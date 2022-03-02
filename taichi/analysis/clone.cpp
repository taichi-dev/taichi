#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"

#include <unordered_map>

TLANG_NAMESPACE_BEGIN

class IRCloner : public IRVisitor {
 private:
  IRNode *other_node;
  std::unordered_map<Stmt *, Stmt *> operand_map_;

 public:
  enum Phase { register_operand_map, replace_operand } phase;

  explicit IRCloner(IRNode *other_node)
      : other_node(other_node), phase(register_operand_map) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Block *stmt_list) override {
    auto other = other_node->as<Block>();
    /*
    for (int i = 0; i < (int)stmt_list->size(); i++) {
      other_node = other->statements[i].get();
      stmt_list->statements[i]->accept(this);
    }
    */
    auto other_head = other->statements.begin();
    for (auto &stmt : stmt_list->statements) {
      other_node = other_head->get();
      stmt->accept(this);
      other_head++;
    }
    other_node = other;
  }

  void generic_visit(Stmt *stmt) {
    if (phase == register_operand_map)
      operand_map_[stmt] = other_node->as<Stmt>();
    else {
      TI_ASSERT(phase == replace_operand);
      auto other_stmt = other_node->as<Stmt>();
      TI_ASSERT(stmt->num_operands() == other_stmt->num_operands());
      for (int i = 0; i < stmt->num_operands(); i++) {
        if (operand_map_.find(stmt->operand(i)) == operand_map_.end())
          other_stmt->set_operand(i, stmt->operand(i));
        else
          other_stmt->set_operand(i, operand_map_[stmt->operand(i)]);
      }
    }
  }

  void visit(Stmt *stmt) override {
    generic_visit(stmt);
  }

  void visit(IfStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<IfStmt>();
    if (stmt->true_statements) {
      other_node = other->true_statements.get();
      stmt->true_statements->accept(this);
      other_node = other;
    }
    if (stmt->false_statements) {
      other_node = other->false_statements.get();
      stmt->false_statements->accept(this);
      other_node = other;
    }
  }

  void visit(WhileStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<WhileStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(RangeForStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<RangeForStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(StructForStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<StructForStmt>();
    other_node = other->body.get();
    stmt->body->accept(this);
    other_node = other;
  }

  void visit(OffloadedStmt *stmt) override {
    generic_visit(stmt);
    auto other = other_node->as<OffloadedStmt>();

#define CLONE_BLOCK(B)                    \
  if (stmt->B) {                          \
    other->B = std::make_unique<Block>(); \
    other_node = other->B.get();          \
    stmt->B->accept(this);                \
  }

    CLONE_BLOCK(tls_prologue)
    CLONE_BLOCK(bls_prologue)
    CLONE_BLOCK(mesh_prologue)

    if (stmt->body) {
      other_node = other->body.get();
      stmt->body->accept(this);
    }

    CLONE_BLOCK(bls_epilogue)
    CLONE_BLOCK(tls_epilogue)
#undef CLONE_BLOCK

    other_node = other;
  }

  static std::unique_ptr<IRNode> run(IRNode *root, Kernel *kernel) {
    std::unique_ptr<IRNode> new_root = root->clone();
    IRCloner cloner(new_root.get());
    cloner.phase = IRCloner::register_operand_map;
    root->accept(&cloner);
    cloner.phase = IRCloner::replace_operand;
    root->accept(&cloner);

    if (kernel != nullptr) {
      new_root->kernel = kernel;
    }
    return new_root;
  }
};

namespace irpass::analysis {
std::unique_ptr<IRNode> clone(IRNode *root, Kernel *kernel) {
  return IRCloner::run(root, kernel);
}
}  // namespace irpass::analysis

TLANG_NAMESPACE_END
