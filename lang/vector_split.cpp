#include "ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockVectorSplit : public IRVisitor {
 public:
  Block *block;
  std::vector<Stmt *> statements;
  std::vector<std::vector<pStmt>> splits;
  int max_width;

  int current_split_factor;
  int lane_start;
  int lane_end;
  std::vector<pStmt> current_split;
  std::unordered_map<Stmt *, std::vector<Stmt *>> origin2split;

  BasicBlockVectorSplit(Block *block) : block(block) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    max_width = 4;
    TC_WARN("max_width set to 4");
    run();
  }

  void run() {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      auto stmt = std::move(block->statements[i]);
      if (stmt->width() > max_width) {
        TC_ASSERT(stmt->width() % max_width == 0);
        current_split_factor = stmt->width() / max_width;
        current_split.resize(current_split_factor);
        lane_start = i * max_width;
        lane_end = (i + 1) * max_width;
        stmt->accept(this);
        origin2split[stmt.get()] =
            std::vector<Stmt *>(current_split_factor, nullptr);
        for (int j = 0; j < current_split_factor; j++) {
          current_split[j]->element_type() = stmt->element_type();
          current_split[j]->width() = max_width;
          origin2split[stmt.get()][j] = current_split[j].get();
        }
        splits.push_back(std::move(current_split));
      } else {
        std::vector<pStmt> split;
        split.push_back(std::move(stmt));
        splits.push_back(std::move(split));
      }
    }
  }

  // Visitors: set current_split[0...current_split_factor]

  void visit(GlobalPtrStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++)
      current_split[i] = Stmt::make<GlobalPtrStmt>(
          stmt->snode.slice(lane_start, lane_end), stmt->indices);
  }

  void visit(ConstStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++)
      current_split[i] =
          Stmt::make<ConstStmt>(stmt->val.slice(lane_start, lane_end));
  }

  void visit(AllocaStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++)
      current_split[i] =
          Stmt::make<AllocaStmt>(max_width, stmt->element_type());
  }

  void visit(LocalLoadStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++)
      current_split[i] =
          Stmt::make<LocalLoadStmt>(stmt->ptr.slice(lane_start, lane_end));
  }

  Stmt *lookup(Stmt *old, int index) {
    TC_ASSERT(origin2split.find(old) != origin2split.end());
    TC_ASSERT(0 <= index);
    TC_ASSERT(index < origin2split[old].size());
    return origin2split[old][index];
  }

  void visit(LocalStoreStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<LocalStoreStmt>(lookup(stmt->ident, i),
                                                    lookup(stmt->stmt, i));
    }
  }
};

// Goal: eliminate vectors that are longer than physical vector width (e.g. 8 on
// AVX2)
class VectorSplit : public IRVisitor {
 public:
  VectorSplit() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Block *stmt_list) {
    std::vector<Stmt *> statements;
    for (auto &stmt : stmt_list->statements) {
      statements.push_back(stmt.get());
    }
    for (auto stmt : statements) {
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

  void visit(RangeForStmt *for_stmt) {
    auto old_vectorize = for_stmt->vectorize;
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  static void run(IRNode *node) {
    VectorSplit inst;
    node->accept(&inst);
  }
};

namespace irpass {

void vector_split(IRNode *root) {
  return VectorSplit::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END