#include "../ir.h"
#include <deque>
#include <set>

TLANG_NAMESPACE_BEGIN

// Lower GlobalPtrStmt into smaller pieces for access optimization

class LowerAccess : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  bool lower_atomic_ptr;
  LowerAccess(bool lower_atomic_ptr) : lower_atomic_ptr(lower_atomic_ptr) {
    // TODO: change this to false
    allow_undefined_visitor = true;
    current_struct_for = nullptr;
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
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
  }

  void lower_scalar_ptr(VecStatement &lowered,
                        SNode *snode,
                        std::vector<Stmt *> indices,
                        bool activate) {
    // emit a sequence of micro access ops
    std::set<SNode *> nodes_on_loop;
    if (current_struct_for) {
      for (SNode *s = current_struct_for->snode; s != nullptr; s = s->parent) {
        nodes_on_loop.insert(s);
      }
    }

    std::deque<SNode *> snodes;
    for (; snode != nullptr; snode = snode->parent)
      snodes.push_front(snode);

    Stmt *last = lowered.push_back<GetRootStmt>();
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

      bool on_loop_tree = nodes_on_loop.find(snode) != nodes_on_loop.end();
      if (on_loop_tree &&
          indices.size() == current_struct_for->loop_vars.size()) {
        for (int j = 0; j < (int)indices.size(); j++) {
          auto diff = analysis::value_diff(indices[j], 0,
                                           current_struct_for->loop_vars[j]);
          if (!diff.linear_related())
            on_loop_tree = false;
          else if (j == (int)indices.size() - 1) {
            if (!(0 <= diff.low &&
                  diff.high <= current_struct_for->vectorize)) {
              on_loop_tree = false;
            }
          } else {
            if (!diff.certain() || diff.low != 0) {
              on_loop_tree = false;
            }
          }
        }
      }

      // linearize
      auto linearized = Stmt::make<LinearizeStmt>(lowered_indices, strides);

      int chid = snode->child_id(snodes[i + 1]);
      auto lookup = Stmt::make<SNodeLookupStmt>(
          snode, last, linearized.get(),
          snode->need_activation() && activate && !on_loop_tree,
          indices);  // if snode has no possibility of null child, set activate
      // = false
      auto get_ch = Stmt::make<GetChStmt>(lookup.get(), chid);

      lowered.push_back(std::move(linearized));
      last = get_ch.get();
      lowered.push_back(std::move(lookup));
      lowered.push_back(std::move(get_ch));
    }
  }

  VecStatement lower_vector_ptr(GlobalPtrStmt *ptr, bool activate) {
    VecStatement lowered;
    std::vector<Stmt *> lowered_pointers;
    for (int i = 0; i < ptr->width(); i++) {
      std::vector<Stmt *> indices;
      for (int k = 0; k < (int)ptr->indices.size(); k++) {
        auto extractor =
            Stmt::make<ElementShuffleStmt>(VectorElement(ptr->indices[k], i));
        indices.push_back(extractor.get());
        lowered.push_back(std::move(extractor));
      }
      lower_scalar_ptr(lowered, ptr->snodes[i], indices, activate);
      TC_ASSERT(lowered.size());
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

  void visit(GlobalLoadStmt *stmt) override {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      auto lowered = lower_vector_ptr(stmt->ptr->as<GlobalPtrStmt>(), false);
      stmt->ptr = lowered.back().get();
      stmt->parent->insert_before(stmt, std::move(lowered));
      throw IRModified();
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      auto lowered = lower_vector_ptr(stmt->ptr->as<GlobalPtrStmt>(), true);
      stmt->ptr = lowered.back().get();
      stmt->parent->insert_before(stmt, std::move(lowered));
      throw IRModified();
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      auto lowered = lower_vector_ptr(stmt->ptr->as<GlobalPtrStmt>(), true);
      stmt->ptr = lowered.back().get();
      stmt->parent->insert_before(stmt, std::move(lowered));
      throw IRModified();
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    if (!lower_atomic_ptr)
      return;
    if (stmt->dest->is<GlobalPtrStmt>()) {
      auto lowered = lower_vector_ptr(stmt->dest->as<GlobalPtrStmt>(), true);
      stmt->dest = lowered.back().get();
      stmt->parent->insert_before(stmt, std::move(lowered));
      throw IRModified();
    }
  }

  static void run(IRNode *node, bool lower_atomic) {
    LowerAccess inst(lower_atomic);
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

void lower_access(IRNode *root, bool lower_atomic) {
  return LowerAccess::run(root, lower_atomic);
}

}  // namespace irpass

TLANG_NAMESPACE_END
