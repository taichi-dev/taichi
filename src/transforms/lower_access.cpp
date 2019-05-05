#include "../ir.h"
#include <deque>

TLANG_NAMESPACE_BEGIN

class LowerAccess : public IRVisitor {
 public:
  LowerAccess() {
    // TODO: change this to false
    allow_undefined_visitor = true;
  }

  void visit(Block *stmt_list) override {
    auto backup_block = current_block;
    current_block = stmt_list;
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
    current_block = backup_block;
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(GlobalPtrStmt *ptr) override {
    if (ptr->width() != 1) {
      TC_WARN("Not lowering vectorized GlobalPtr");
      return;
    }
    // emit a sequence of micro access ops
    std::deque<SNode *> snodes;
    for (auto snode = ptr->snodes[0]; snode != nullptr; snode = snode->parent)
      snodes.push_front(snode);
    Stmt *last = nullptr;
    VecStatement lowered;
    for (int i = 0; i < (int)snodes.size() - 1; i++) {
      auto snode = snodes[i];
      std::vector<Stmt *> lowered_indices;
      std::vector<int> strides;
      std::vector<int> offsets;
      // extract bits
      for (int k_ = 0; k_ < (int)ptr->indices.size(); k_++) {
        for (int k = 0; k < max_num_indices; k++) {
          if (snode->physical_index_position[k_] == k) {
            int begin = snode->extractors[k].start;
            int end = begin + snode->extractors[k].num_bits;
            auto extracted = Stmt::make<OffsetAndExtractBitsStmt>(
                ptr->indices[k_], begin, end, 0);
            lowered_indices.push_back(extracted.get());
            lowered.push_back(std::move(extracted));
            strides.push_back(1 << snode->extractors[k].num_bits);
            offsets.push_back(0);
          }
        }
      }

      // linearize
      auto linearized =
          Stmt::make<LinearizeStmt>(lowered_indices, strides, offsets);

      auto lookup = Stmt::make<SNodeLookupStmt>(
          snode, snode->child_id(snodes[i + 1]), last, linearized.get(), true,
          ptr->indices);

      lowered.push_back(std::move(linearized));
      last = lookup.get();
      lowered.push_back(std::move(lookup));
    }
    ptr->parent->replace_with(ptr, lowered);
    throw IRModifiedException();
  }

  static void run(IRNode *node) {
    LowerAccess inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (IRModifiedException) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void lower_access(IRNode *root) {
  return LowerAccess::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END