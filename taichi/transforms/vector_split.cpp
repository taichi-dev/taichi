// Split vectors wider than machine vector width into multiple vectors

#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockVectorSplit : public IRVisitor {
 public:
  Block *block;
  std::vector<Stmt *> statements;
  std::vector<std::vector<pStmt>> splits;
  int max_width;

  int current_split_factor;
  std::vector<pStmt> current_split;
  bool need_split;
  bool serial_schedule;
  std::unordered_map<Stmt *, std::vector<Stmt *>> origin2split;

  BasicBlockVectorSplit(Block *block, int max_width, bool serial_schedule)
      : block(block), max_width(max_width), serial_schedule(serial_schedule) {
    // allow_undefined_visitor = true;
    // invoke_default_visitor = false;
    run();
  }

  int lane_start(int split) {
    return split * max_width;
  }

  int lane_end(int split) {
    return (split + 1) * max_width;
  }

  Stmt *lookup(Stmt *old, int index) {
    if (origin2split.find(old) == origin2split.end()) {
      TI_WARN("VectorSplitter looking for statement outside current block?");
      return old;
    } else {
      TI_ASSERT(0 <= index);
      TI_ASSERT(index < (int)origin2split[old].size());
      return origin2split[old][index];
    }
  }

  void run() {
    std::vector<pStmt> statements = std::move(block->statements);
    for (int i = 0; i < (int)statements.size(); i++) {
      auto stmt = statements[i].get();
      if (stmt->width() > max_width) {
        TI_ASSERT(stmt->width() % max_width == 0);
        current_split_factor = stmt->width() / max_width;
        current_split.resize(current_split_factor);
        need_split = true;
        stmt->accept(this);
        origin2split[stmt] = std::vector<Stmt *>(current_split_factor, nullptr);
        for (int j = 0; j < current_split_factor; j++) {
          current_split[j]->element_type() = stmt->element_type();
          current_split[j]->width() = max_width;
          origin2split[stmt][j] = current_split[j].get();
        }
        splits.push_back(std::move(current_split));
      } else {  // recreate a statement anyway since the original one may be
        // pointing to unknown statements
        current_split_factor = 1;
        current_split.resize(current_split_factor);
        need_split = false;
        stmt->accept(this);
        origin2split[stmt] = std::vector<Stmt *>(1, nullptr);
        current_split[0]->width() = stmt->width();
        current_split[0]->element_type() = stmt->element_type();
        origin2split[stmt][0] = current_split[0].get();
        std::vector<pStmt> split;
        split.push_back(std::move(current_split[0]));
        splits.push_back(std::move(split));
      }
    }
    block->statements.clear();
    if (!serial_schedule) {
      // finish vectors one by one
      for (int i = 0; i < (int)splits.size(); i++) {
        for (int j = 0;; j++) {
          bool modified = false;
          if (j < (int)splits[i].size()) {
            block->insert(std::move(splits[i][j]));
            modified = true;
          }
          if (!modified) {
            break;
          }
        }
      }
    } else {
      for (int j = 0;; j++) {
        bool modified = false;
        for (int i = 0; i < (int)splits.size(); i++) {
          if (j < (int)splits[i].size()) {
            block->insert(std::move(splits[i][j]));
            modified = true;
          }
        }
        if (!modified) {
          break;
        }
      }
    }
    for (int i = 0; i < (int)block->statements.size(); i++) {
      auto stmt_ = block->statements[i].get();
      if (stmt_->is<LocalLoadStmt>()) {
        auto stmt = stmt_->as<LocalLoadStmt>();
        for (int l = 0; l < stmt->width(); l++) {
          auto *old_var = stmt->ptr[l].var;
          if (origin2split.find(old_var) != origin2split.end()) {
            auto new_var =
                origin2split[old_var][stmt->ptr[l].offset / max_width];
            stmt->ptr[l].var = new_var;
            stmt->ptr[l].offset %= max_width;
            // TI_WARN("replaced...");
          }
        }
      }
    }
  }

  // Visitors: set current_split[0...current_split_factor]

  void visit(GlobalPtrStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      std::vector<Stmt *> indices;
      for (int j = 0; j < (int)stmt->indices.size(); j++) {
        indices.push_back(lookup(stmt->indices[j], i));
      }
      current_split[i] = Stmt::make<GlobalPtrStmt>(
          stmt->snodes.slice(lane_start(i),
                             need_split ? lane_end(i) : stmt->width()),
          indices);
    }
  }

  void visit(ConstStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<ConstStmt>(stmt->val.slice(
          lane_start(i), need_split ? lane_end(i) : stmt->width()));
    }
  }

  void visit(AllocaStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++)
      current_split[i] = Stmt::make<AllocaStmt>(
          need_split ? max_width : stmt->width(), stmt->element_type());
  }

  void visit(ElementShuffleStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      LaneAttribute<VectorElement> ptr;
      int new_width = need_split ? max_width : stmt->width();
      ptr.resize(new_width);
      for (int j = 0; j < new_width; j++) {
        VectorElement addr(stmt->elements[lane_start(i) + j]);
        if (origin2split.find(addr.stmt) == origin2split.end()) {
          ptr[j] = addr;
        } else {
          ptr[j].stmt = lookup(addr.stmt, addr.index / max_width);
          ptr[j].index = addr.index % max_width;
        }
      }
      current_split[i] = Stmt::make<ElementShuffleStmt>(ptr);
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      LaneAttribute<LocalAddress> ptr;
      int new_width = need_split ? max_width : stmt->width();
      ptr.reserve(new_width);
      for (int j = 0; j < new_width; j++) {
        LocalAddress addr(stmt->ptr[lane_start(i) + j]);
        if (origin2split.find(addr.var) == origin2split.end()) {
          ptr.push_back(addr);
        } else {
          ptr.push_back(LocalAddress(lookup(addr.var, addr.offset / max_width),
                                     addr.offset % max_width));
        }
      }
      current_split[i] = Stmt::make<LocalLoadStmt>(ptr);
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<LocalStoreStmt>(lookup(stmt->ptr, i),
                                                    lookup(stmt->data, i));
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<GlobalLoadStmt>(lookup(stmt->ptr, i));
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<GlobalStoreStmt>(lookup(stmt->ptr, i),
                                                     lookup(stmt->data, i));
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] =
          Stmt::make<UnaryOpStmt>(stmt->op_type, lookup(stmt->operand, i));
      current_split[i]->as<UnaryOpStmt>()->cast_type =
          stmt->as<UnaryOpStmt>()->cast_type;
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<BinaryOpStmt>(
          stmt->op_type, lookup(stmt->lhs, i), lookup(stmt->rhs, i));
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] =
          Stmt::make<TernaryOpStmt>(stmt->op_type, lookup(stmt->op1, i),
                                    lookup(stmt->op2, i), lookup(stmt->op3, i));
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<AtomicOpStmt>(
          stmt->op_type, lookup(stmt->dest, i), lookup(stmt->val, i));
    }
  }

  void visit(PrintStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] =
          Stmt::make<PrintStmt>(lookup(stmt->stmt, i), stmt->str);
    }
  }

  void visit(RandStmt *stmt) override {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<RandStmt>(stmt->element_type());
    }
  }

  void visit(WhileControlStmt *stmt) override {
    TI_ASSERT(need_split == false);
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<WhileControlStmt>(lookup(stmt->mask, i),
                                                      lookup(stmt->cond, i));
    }
  }
};

// Goal: eliminate vectors that are longer than physical vector width (e.g. 8
// on AVX2)
class VectorSplit : public IRVisitor {
 public:
  int max_width;
  bool serial_schedule;

  VectorSplit(IRNode *node, int max_width, bool serial_schedule)
      : max_width(max_width), serial_schedule(serial_schedule) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);
  }

  void visit(Block *block) override {
    if (!block->has_container_statements()) {
      bool all_within_width = true;
      for (auto &stmt : block->statements) {
        if (stmt->width() > max_width) {
          all_within_width = false;
        }
      }
      if (!all_within_width)
        BasicBlockVectorSplit(block, max_width, serial_schedule);
    } else {
      for (auto &stmt : block->statements) {
        stmt->accept(this);
      }
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
};

namespace irpass {

void vector_split(IRNode *root, int max_width, bool serial_schedule) {
  VectorSplit(root, max_width, serial_schedule);
}

}  // namespace irpass

TLANG_NAMESPACE_END
