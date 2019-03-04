#include <typeinfo>
#include <set>
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockSLP : public IRVisitor {
 public:
  Block *block;
  std::set<Stmt *> inside;
  std::set<Stmt *> visited;
  int width;
  using Pack = std::vector<Stmt *>;
  std::vector<std::pair<Pack, Stmt *>> existing_stmts;
  VecStatement new_stmts;

  Pack tmp_operands;
  std::unique_ptr<Stmt> tmp_stmt;
  Pack building_pack;

  BasicBlockSLP() {
    // allow_undefined_visitor = true;
    // invoke_default_visitor = true;
  }

  void update_type(Statement *stmt) {
    tmp_stmt->ret_type = stmt->ret_type;
    tmp_stmt->ret_type.width *= width;
  }

  void visit(ConstStmt *stmt) override {
    tmp_stmt = std::make_unique<ConstStmt>(0);
    update_type(stmt);
  }

  void visit(AllocaStmt *stmt) override {
    tmp_stmt = std::make_unique<AllocaStmt>(stmt->ret_type.data_type);
    update_type(stmt);
  }

  void visit(LocalLoadStmt *stmt) override {
    LaneAttribute<LocalAddress> ptr;
    for (int i = 0; i < width; i++) {
      ptr += dynamic_cast<LocalLoadStmt *>(building_pack[i])->ptr;
    }
    tmp_stmt = std::make_unique<LocalLoadStmt>(ptr);
    tmp_stmt->ret_type.width = stmt->ret_type.width * width;
    update_type(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    tmp_stmt =
        std::make_unique<LocalStoreStmt>(tmp_operands[0], tmp_operands[1]);
    tmp_stmt->ret_type.width = stmt->ret_type.width * width;
    update_type(stmt);
  }

  void visit(GlobalPtrStmt *stmt) override {
    std::vector<Stmt *> indices = tmp_operands;
    TC_TAG;
    LaneAttribute<SNode *> snodes;

    TC_TAG;
    for (int i = 0; i < width; i++) {
      TC_P(building_pack[i]->as<GlobalPtrStmt>()->snode[0]->node_type_name);
      snodes += building_pack[i]->as<GlobalPtrStmt>()->snode;
    }
    TC_TAG;

    TC_P(snodes.size());
    for (int i = 0; i < snodes.size(); i++) {
      TC_P(snodes[i]);
    }
    tmp_stmt = Stmt::make<GlobalPtrStmt>(snodes, indices);
    TC_TAG;
    tmp_stmt->ret_type.width = stmt->ret_type.width * width;
    TC_TAG;
    update_type(stmt);
    TC_TAG;
  }

  void visit(GlobalStoreStmt *stmt) override {
    tmp_stmt = Stmt::make<GlobalStoreStmt>(tmp_operands[0], tmp_operands[1]);
    update_type(stmt);
  }

  // merge tmp_operands into one statement
  void visit(BinaryOpStmt *stmt) override {
    tmp_stmt = std::make_unique<BinaryOpStmt>(
        dynamic_cast<BinaryOpStmt *>(building_pack[0])->op_type,
        tmp_operands[0], tmp_operands[1]);
    update_type(stmt);
  }

  Stmt *find_stmt(const Pack &pack) {
    TC_ASSERT((int)pack.size() == width);
    for (int i = 0; i < (int)existing_stmts.size(); i++) {
      bool match = true;
      for (int j = 0; j < width; j++) {
        if (existing_stmts[i].first[j] != pack[j]) {
          match = false;
        }
      }
      if (match) {
        return existing_stmts[i].second;
      }
    }
    return nullptr;
  }

  // create a new stmt out of the pack
  Stmt *build(const Pack &pack) {
    for (int i = 0; i < width; i++) {
      fmt::print("{} ", pack[i]->id);
    }
    fmt::print("\n");
    auto existing = find_stmt(pack);
    if (existing) {
      return existing;
    }
    Pack operands;
    if (!pack[0]->is<LocalLoadStmt>()) {
      for (int i = 0; i < (int)pack[0]->operands.size(); i++) {
        Pack operand_pack;
        for (int j = 0; j < (int)pack.size(); j++) {
          operand_pack.push_back(*pack[j]->operands[i]);
        }
        operands.push_back(build(operand_pack));
      }
    }
    tmp_operands = operands;
    building_pack = pack;
    TC_ASSERT(tmp_stmt == nullptr);
    pack[0]->accept(this);
    TC_ASSERT(tmp_stmt != nullptr);
    tmp_operands.clear();
    Stmt *ret = tmp_stmt.get();
    new_stmts.push_back(std::move(tmp_stmt));
    return ret;
  }

  // replace with BBlock with SLP'ed block
  void run(Block *block, int width) {
    this->width = width;
    visited.clear();
    std::vector<std::unique_ptr<Stmt>> stmts = std::move(block->statements);
    // Find the last statement
    auto last_stmt = stmts.back().get();

    for (auto &stmt : block->statements) {
      inside.insert(stmt.get());
    }

    std::vector<Stmt *> seed_statements;

    seed_statements.push_back(last_stmt);

    // from the back, find the other (width - 1) statements of the same type
    for (int i = 0; i < (int)stmts.size() - 1; i++) {
      if (typeid(*last_stmt) == typeid(*stmts[i])) {
        // found a stmt of the same type.
        seed_statements.push_back(stmts[i].get());
        if ((int)seed_statements.size() == width) {
          break;
        }
      }
    }

    if ((int)seed_statements.size() != width) {
      TC_ERROR("Cannot find enough {} seed statements to start SLP search.",
               width);
    }

    build(seed_statements);
    // TODO: check order. SLP should not change order of local/global
    // sort the statements...
    // load/store...
    TC_TAG;
    block->set_statements(std::move(new_stmts));
  }
};

class SLPVectorize : public IRVisitor {
 public:
  SLPVectorize() {
    allow_undefined_visitor = true;
    // invoke_default_visitor = true;
  }

  void visit(Block *block) {
    if (block->slp != 1) {
      auto slp = BasicBlockSLP();
      slp.run(block, block->slp);
      block->slp = 1;
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
    for_stmt->body->accept(this);
  }

  void visit(WhileStmt *stmt) {
    stmt->body->accept(this);
  }

  static void run(IRNode *node) {
    SLPVectorize inst;
    node->accept(&inst);
  }
};

namespace irpass {

void slp_vectorize(IRNode *root) {
  return SLPVectorize::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
