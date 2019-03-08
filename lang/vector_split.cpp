#include "ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockVectorSplit : public IRVisitor {
 public:
  Block *block;
  std::vector<Stmt *> statements;
  std::vector<std::vector<pStmt>> splits;
  int max_width;

  int current_split_factor;
  std::vector<pStmt> current_split;
  std::unordered_map<Stmt *, std::vector<Stmt *>> origin2split;

  BasicBlockVectorSplit(Block *block, int max_width)
      : block(block), max_width(max_width) {
    // allow_undefined_visitor = true;
    // invoke_default_visitor = false;
    TC_WARN("max_width set to 4");
    run();
  }

  int lane_start(int split) {
    return split * max_width;
  }

  int lane_end(int split) {
    return (split + 1) * max_width;
  }

  Stmt *lookup(Stmt *old, int index) {
    TC_ASSERT(origin2split.find(old) != origin2split.end());
    TC_ASSERT(0 <= index);
    TC_ASSERT(index < origin2split[old].size());
    return origin2split[old][index];
  }

  void run() {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      auto stmt = std::move(block->statements[i]);
      if (stmt->width() > max_width) {
        TC_ASSERT(stmt->width() % max_width == 0);
        current_split_factor = stmt->width() / max_width;
        current_split.resize(current_split_factor);
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
    block->statements.clear();
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

  // Visitors: set current_split[0...current_split_factor]

  void visit(GlobalPtrStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      std::vector<Stmt *> indices;
      for (int j = 0; j < stmt->indices.size(); j++) {
        indices.push_back(lookup(stmt->indices[j], i));
      }
      current_split[i] = Stmt::make<GlobalPtrStmt>(
          stmt->snode.slice(lane_start(i), lane_end(i)), indices);
    }
  }

  void visit(ConstStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] =
          Stmt::make<ConstStmt>(stmt->val.slice(lane_start(i), lane_end(i)));
    }
  }

  void visit(AllocaStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++)
      current_split[i] =
          Stmt::make<AllocaStmt>(max_width, stmt->element_type());
  }

  void visit(LocalLoadStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      LaneAttribute<LocalAddress> ptr;
      ptr.resize(max_width);
      for (int j = 0; j < max_width; j++) {
        LocalAddress addr(stmt->ptr[lane_start(i) + j]);
        if (origin2split.find(addr.var) == origin2split.end()) {
          ptr[j] = addr;
        } else {
          ptr[j].var = lookup(addr.var, addr.offset / max_width);
          ptr[j].offset = addr.offset % max_width;
        }
      }
      current_split[i] = Stmt::make<LocalLoadStmt>(ptr);
    }
  }

  void visit(LocalStoreStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<LocalStoreStmt>(lookup(stmt->ptr, i),
                                                    lookup(stmt->data, i));
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<GlobalLoadStmt>(lookup(stmt->ptr, i));
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<GlobalStoreStmt>(lookup(stmt->ptr, i),
                                                     lookup(stmt->data, i));
    }
  }

  void visit(BinaryOpStmt *stmt) {
    for (int i = 0; i < current_split_factor; i++) {
      current_split[i] = Stmt::make<BinaryOpStmt>(
          stmt->op_type, lookup(stmt->lhs, i), lookup(stmt->rhs, i));
    }
  }
};

// Goal: eliminate vectors that are longer than physical vector width (e.g. 8 on
// AVX2)
class VectorSplit : public IRVisitor {
 public:
  int max_width;

  VectorSplit(IRNode *node, int max_width) : max_width(max_width) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
    node->accept(this);
  }

  void visit(Block *block) {
    if (!block->has_container_statements()) {
      BasicBlockVectorSplit(block, max_width);
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

  void visit(RangeForStmt *for_stmt) {
    auto old_vectorize = for_stmt->vectorize;
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }
};

namespace irpass {

void vector_split(IRNode *root, int max_width) {
  VectorSplit(root, max_width);
}

}  // namespace irpass

TLANG_NAMESPACE_END