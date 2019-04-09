#include "../ir.h"
#include "../scratch_pad.h"

TLANG_NAMESPACE_BEGIN

// Figure out accessed snodes, and their ranges in this for stmt
class AccessAnalysis : public IRVisitor {
 public:
  StructForStmt *for_stmt;
  ScratchPads *pads;

  std::vector<std::vector<int>> block_indices;

  AccessAnalysis(StructForStmt *for_stmt, ScratchPads *pads)
      : for_stmt(for_stmt) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;

    generate_block_indices(for_stmt->snode->parent, {}, 0);

    /*
    TC_P(block_indices.size());
    for (int i = 0; i < block_indices.size(); i++) {
      TC_P(block_indices[i]);
    }
    */

    const auto &block = for_stmt->body;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      block->statements[i]->accept(this);
    }

    pads->print();
  }

  void generate_block_indices(SNode *snode, std::vector<int> index, int s) {
    // NOTE: Assuming not vectorized
    if (s == max_num_indices) {
      block_indices.push_back(index);
      return;
    }

    if (snode->extractors[s].active) {
      for (int i = 0; i < (1 << snode->extractors[s].num_bits); i++) {
        auto new_index = index;
        new_index.push_back(i);
        generate_block_indices(snode, new_index, s + 1);
      }
    } else {
      generate_block_indices(snode, index, s + 1);
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override {
    TC_ASSERT(stmt->width() == 1);  // TODO: support vectorization
    for (int l = 0; l < stmt->width(); l++) {
      auto ptr_ = stmt->ptr;
      auto ptr = ptr_->as<GlobalPtrStmt>();
      auto snode = ptr->snodes[l];
      bool matching_indices = true;
      std::vector<int> offsets;  // TODO: change to offset_ranges
      offsets.resize(ptr->indices.size());
      int num_indices = (int)ptr->indices.size();
      for (int i = 0; i < num_indices; i++) {
        auto diff =
            analysis::value_diff(ptr->indices[i], l, for_stmt->loop_vars[i]);
        if (diff.first) {
          offsets[i] = diff.second;
        } else {
          matching_indices = false;
        }
      }
      if (matching_indices) {
        TC_INFO("Detected regular access");
        for (int i = 0; i < num_indices; i++) {
          TC_P(offsets[i]);
        }
        for (const auto &bind : block_indices) {
          auto access_ind = bind;
          for (int d = 0; d < num_indices; d++) {
            access_ind[d] += offsets[d];
          }
          // TC_P(access_ind);
          pads->access(snode, access_ind, ScratchPad::AccessFlag::read);
        }
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    return;
  }

  void visit(AtomicOpStmt *stmt) override {
    return;
  }

  void visit(Stmt *stmt) override {
    TC_ASSERT(!stmt->is_container_statement());
  }
};

class InsertScratchPad : public IRVisitor {
 public:
  std::unique_ptr<ScratchPads> pads;
  InsertScratchPad(StructForStmt *node) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);
  }

  void visit(Block *block) override {
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
  }

  void visit(RangeForStmt *for_stmt) override {
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) override {
    stmt->body->accept(this);
  }

  void visit(StructForStmt *for_stmt) override {
    // do the work here...
    TC_P(for_stmt->cached_level);
    if (for_stmt->cached_level != -1) {
      AccessAnalysis _(for_stmt, pads.get());
      // WeakenAccess _(for_stmt);
    }
    for_stmt->body->accept(this);
  }

  std::unique_ptr<ScratchPads> get() {
    return std::move(pads);
  }
};

namespace irpass {

std::unique_ptr<ScratchPads> initialize_scratch_pad(StructForStmt *root) {
  InsertScratchPad _(root);
  return _.get();
}

}  // namespace irpass

TLANG_NAMESPACE_END
