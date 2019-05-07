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

  void lower_scalar_ptr(VecStatement &lowered,
                        SNode *snode,
                        std::vector<Stmt *> indices,
                        bool activate) {
    // emit a sequence of micro access ops
    std::deque<SNode *> snodes;
    for (; snode != nullptr; snode = snode->parent)
      snodes.push_front(snode);
    Stmt *last = nullptr;
    for (int i = 0; i < (int)snodes.size() - 1; i++) {
      auto snode = snodes[i];
      std::vector<Stmt *> lowered_indices;
      std::vector<int> strides;
      // extract bits
      for (int k_ = 0; k_ < (int)indices.size(); k_++) {
        for (int k = 0; k < max_num_indices; k++) {
          if (snode->physical_index_position[k_] == k) {
            int begin = snode->extractors[k].start;
            int end = begin + snode->extractors[k].num_bits;
            auto extracted = Stmt::make<OffsetAndExtractBitsStmt>(
                indices[k_], begin, end, 0);
            lowered_indices.push_back(extracted.get());
            lowered.push_back(std::move(extracted));
            strides.push_back(1 << snode->extractors[k].num_bits);
          }
        }
      }

      // linearize
      auto linearized = Stmt::make<LinearizeStmt>(lowered_indices, strides);

      auto lookup = Stmt::make<SNodeLookupStmt>(
          snode, snode->child_id(snodes[i + 1]), last, linearized.get(),
          snode->has_null() && activate,
          indices);  // if snode has no possibility of null child, set activate
                     // = false

      lowered.push_back(std::move(linearized));
      last = lookup.get();
      lowered.push_back(std::move(lookup));
    }
  }

  VecStatement lower_vector_ptr(GlobalPtrStmt *ptr, bool activate) {
    VecStatement lowered;
    std::vector<Stmt *> lowered_pointers;
    for (int i = 0; i < ptr->width(); i++) {
      std::vector<Stmt *> indices;
      for (int k = 0; k < ptr->indices.size(); k++) {
        auto extractor =
            Stmt::make<ElementShuffleStmt>(VectorElement(ptr->indices[k], i));
        indices.push_back(extractor.get());
        lowered.push_back(std::move(extractor));
      }
      lower_scalar_ptr(lowered, ptr->snodes[i], indices, activate);
      lowered_pointers.push_back(lowered.back().get());
    }
    // create shuffle
    LaneAttribute<VectorElement> lanes;
    for (int i = 0; i < ptr->width(); i++) {
      lanes.push_back(VectorElement(lowered_pointers[i], 0));
    }
    auto merge = Stmt::make<ElementShuffleStmt>(lanes, true);
    merge->ret_type.data_type = ptr->snodes[0]->dt;
    lowered.push_back(std::move(merge));
    return lowered;
  }

  void visit(GlobalLoadStmt *stmt) {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      auto lowered = lower_vector_ptr(stmt->ptr->as<GlobalPtrStmt>(), false);
      stmt->ptr = lowered.back().get();
      stmt->parent->insert_before(stmt, lowered);
      throw IRModified();
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      auto lowered = lower_vector_ptr(stmt->ptr->as<GlobalPtrStmt>(), true);
      stmt->ptr = lowered.back().get();
      stmt->parent->insert_before(stmt, lowered);
      throw IRModified();
    }
  }

  static void run(IRNode *node) {
    LowerAccess inst;
    while (true) {
      bool modified = false;
      try {
        node->accept(&inst);
      } catch (IRModified) {
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