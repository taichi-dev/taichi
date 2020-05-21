#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/scratch_pad.h"

TLANG_NAMESPACE_BEGIN

// Figure out accessed snodes, and their ranges in this for stmt
class AccessAnalysis : public IRVisitor {
 public:
  StructForStmt *for_stmt;
  ScratchPads *pads;

  std::vector<std::vector<int>> block_indices;

  AccessAnalysis(StructForStmt *for_stmt, ScratchPads *pads)
      : for_stmt(for_stmt), pads(pads) {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;

    TI_WARN(
        "Using the size of scratch_opt[0].second as the snode size to cache");
    generate_block_indices(for_stmt->scratch_opt[0].second->parent, {}, 0);

    const auto &block = for_stmt->body;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      block->statements[i]->accept(this);
    }

    pads->print();
  }

  void generate_block_indices(SNode *snode, std::vector<int> index, int s) {
    // NOTE: Assuming not vectorized
    if (s == taichi_max_num_indices) {
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

  void access(Stmt *stmt, AccessFlag flag) {
    auto ptr = stmt->as<GlobalPtrStmt>();
    for (int l = 0; l < stmt->width(); l++) {
      auto snode = ptr->snodes[l];
      // std::vector<SNode *> snodes;
      /*
      for (auto it: pads->pads) {
        //snodes.push_back(it.first);
        TI_P(it.first->node_type_name);
      }
      TI_P(snode->node_type_name);
      */
      if (!pads->has(snode)) {
        continue;
      }
      bool matching_indices = true;
      std::vector<std::pair<int, int>> offsets;
      offsets.resize(ptr->indices.size());
      int num_indices = (int)ptr->indices.size();
      for (int i = 0; i < num_indices; i++) {
        auto diff = irpass::analysis::value_diff(ptr->indices[i], l,
                                                 for_stmt->loop_vars[i]);
        if (diff.linear_related()) {
          offsets[i].first = diff.low;
          offsets[i].second = diff.high;
          /*
          TI_P(ptr->name());
          TI_P(diff.low);
          TI_P(diff.high);
          */
        } else {
          /*
          TI_P(i);
          TI_P(for_stmt->loop_vars[i]->raw_name());
          TI_P(ptr->indices[i]->raw_name());
          */
          matching_indices = false;
        }
      }
      if (matching_indices) {
        /*
        TI_INFO("Detected regular access");
        for (int i = 0; i < num_indices; i++) {
          TI_P(offsets[i]);
        }
        */
        for (const auto &bind : block_indices) {
          std::function<void(std::vector<int>, int)> visit =
              [&](std::vector<int> ind, int depth) {
                if (depth == num_indices) {
                  pads->access(snode, ind, flag);
                  return;
                }
                for (int i = offsets[depth].first; i < offsets[depth].second;
                     i++) {
                  ind[depth] = bind[depth] + i;
                  visit(ind, depth + 1);
                }
              };
          visit(bind, 0);
        }
      }
    }
  }

  // Do not eliminate global data access
  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);  // TODO: support vectorization
    access(stmt->ptr, AccessFlag::read);
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);  // TODO: support vectorization
    access(stmt->ptr, AccessFlag::write);
  }

  void visit(AtomicOpStmt *stmt) override {
    if (stmt->op_type == AtomicOpType::add) {
      access(stmt->dest, AccessFlag::accumulate);
    }
  }

  void visit(Stmt *stmt) override {
    TI_ASSERT(!stmt->is_container_statement());
  }
};

class InsertScratchPad : public IRVisitor {
 public:
  std::unique_ptr<ScratchPads> pads;
  InsertScratchPad(StructForStmt *node) {
    pads = std::make_unique<ScratchPads>();
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);
    pads->finalize();
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
    if (!for_stmt->scratch_opt.empty()) {
      for (auto &opt : for_stmt->scratch_opt) {
        pads->insert(opt.second);
      }
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
