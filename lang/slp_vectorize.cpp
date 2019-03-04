#include <typeinfo>
#include <set>
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class BasicBlockSLP : public IRVisitor {
 public:
  Block *block;
  std::set<Stmt *> inside;
  std::set<Stmt *> visited;
  std::vector<pStmt> *input_statements;
  int width;
  using Pack = std::vector<Stmt *>;
  std::vector<std::pair<Pack, Stmt *>> existing_stmts;
  VecStatement new_stmts;

  Pack tmp_operands;
  std::unique_ptr<Stmt> tmp_stmt;
  std::unordered_map<Stmt *, int> position;
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
    LaneAttribute<TypedConstant> val;
    for (int i = 0; i < width; i++) {
      val += dynamic_cast<ConstStmt *>(building_pack[i])->val;
    }
    tmp_stmt = std::make_unique<ConstStmt>(val);
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
    LaneAttribute<SNode *> snodes;
    for (int i = 0; i < width; i++) {
      snodes += building_pack[i]->as<GlobalPtrStmt>()->snode;
    }
    tmp_stmt = Stmt::make<GlobalPtrStmt>(snodes, indices);
    tmp_stmt->ret_type.width = stmt->ret_type.width * width;
    update_type(stmt);
  }

  void visit(GlobalStoreStmt *stmt) override {
    tmp_stmt = Stmt::make<GlobalStoreStmt>(tmp_operands[0], tmp_operands[1]);
    update_type(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    tmp_stmt = Stmt::make<GlobalLoadStmt>(tmp_operands[0]);
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
    auto existing = find_stmt(pack);
    if (existing) {
      return existing;
    }
    for (int i = 0; i < width; i++) {
      fmt::print(" {} ", pack[i]->id);
      TC_ASSERT(visited.find(pack[i]) == visited.end());
      visited.insert(pack[i]);
    }
    fmt::print("\n");
    Pack operands;
    if (!pack[0]->is<LocalLoadStmt>()) {
      for (int i = 0; i < (int)pack[0]->operands.size(); i++) {
        Pack operand_pack;
        for (int j = 0; j < (int)pack.size(); j++) {
          operand_pack.push_back(*pack[j]->operands[i]);
        }
        operands.push_back(build(operand_pack));
      }
    } else {
      TC_ASSERT(pack[0]->width() == 1);
      // Pack previous store or alloca.
      for (int i = 0; i < (int)pack[0]->as<LocalLoadStmt>()->ptr.size(); i++) {
        Pack operand_pack;
        for (int j = 0; j < (int)pack.size(); j++) {
          auto previous = pack[j]->as<LocalLoadStmt>()->previous_store_or_alloca_in_block();
          if (previous)
            operand_pack.push_back(previous);
        }
        if (operand_pack.size() != 0) {
          TC_ASSERT((int)operand_pack.size() == width);
          operands.push_back(build(operand_pack));
        }
      }
    }
    tmp_operands = operands;
    building_pack = pack;
    TC_ASSERT(tmp_stmt == nullptr);
    pack[0]->accept(this);
    TC_ASSERT(tmp_stmt != nullptr);
    tmp_operands.clear();
    auto ret = new_stmts.push_back(std::move(tmp_stmt));

    int pos = -1;
    for (int i = 0; i < (int)input_statements->size(); i++) {
      if (pack[0] == (*input_statements)[i].get()) {
        pos = i;
        break;
      }
    }
    TC_ASSERT(pos != -1);
    position[ret] = pos;
    existing_stmts.push_back(std::make_pair(pack, ret));
    return ret;
  }

  // replace with BBlock with SLP'ed block
  void run(Block *block, int width) {
    this->block = block;
    this->width = width;
    visited.clear();
    input_statements = &block->statements;
    auto &stmts = *input_statements;
    while (1) {
      TC_INFO("Seeding...");
      // Find the last statement
      Stmt *last_stmt = nullptr;
      for (int i = stmts.size() - 1; i >= 0; i--) {
        if (visited.find(stmts[i].get()) == visited.end()) {
          last_stmt = stmts[i].get();
          break;
        }
      }
      if (last_stmt == nullptr) {
        break;
      }

      std::vector<Stmt *> seed_statements;

      // from the back, find the other (width - 1) statements of the same type
      for (int i = 0; i < (int)stmts.size(); i++) {
        if (typeid(*last_stmt) == typeid(*stmts[i])) {
          // found a stmt of the same type.
          seed_statements.push_back(stmts[i].get());
        }
        if ((int)seed_statements.size() == width) {
          break;
        }
      }

      if ((int)seed_statements.size() != width) {
        TC_ERROR("Cannot find enough {} seed statements to start SLP search.",
                 width);
      }

      build(seed_statements);
    }
    sort(new_stmts);
    fix_alloca_ref(new_stmts.stmts);
    block->set_statements(std::move(new_stmts));
  }

  void fix_alloca_ref(std::vector<pStmt> &stmts) {
    for (auto &stmt_ : stmts) {
      if (stmt_->is<LocalLoadStmt>()) {
        auto stmt = stmt_->as<LocalLoadStmt>();
        for (int i = 0; i < (int)stmt->ptr.size(); i++) {
          auto old_stmt = stmt->ptr[i].var;
          if (visited.find(old_stmt) != visited.end()) {
            bool replaced = false;
            // replace with packed alloca...
            for (auto &rec : existing_stmts) {
              for (int j = 0; j < width; j++) {
                if (rec.first[j] == old_stmt) {
                  // replace alloca
                  stmt->ptr[i].var = rec.second;
                  // compute new offset
                  stmt->ptr[i].offset += j * old_stmt->width();
                  replaced = true;
                  break;
                }
              }
              if (replaced)
                break;
            }
            TC_ASSERT(replaced);
          }
        }
      }
    }
  }

  void sort(VecStatement &vec) {
    std::sort(vec.stmts.begin(), vec.stmts.end(),
              [&](const pStmt &a, const pStmt &b) {
                return position[a.get()] < position[b.get()];
              });
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
