#include <typeinfo>
#include <set>
#include "ir.h"

TLANG_NAMESPACE_BEGIN

using PackRecord = std::unordered_map<Stmt *, std::pair<Stmt *, int>>;

class BasicBlockSLP : public IRVisitor {
 public:
  Block *block;
  PackRecord *rec;
  std::set<Stmt *> inside;
  std::set<Stmt *> visited;
  std::vector<Stmt *> *input_statements;
  int slp_width;
  using Pack = std::vector<Stmt *>;
  std::vector<std::pair<Pack, Stmt *>> existing_stmts;
  VecStatement new_stmts;

  Pack tmp_operands;
  std::unique_ptr<Stmt> tmp_stmt;
  std::unordered_map<Stmt *, int> position;
  Pack building_pack;
  std::vector<pStmt> shuffles;

  BasicBlockSLP() {
    // allow_undefined_visitor = true;
    // invoke_default_visitor = true;
  }
  void update_type(Statement *stmt) {
    tmp_stmt->ret_type = stmt->ret_type;
    tmp_stmt->ret_type.width *= slp_width;
  }

  void visit(ConstStmt *stmt) override {
    LaneAttribute<TypedConstant> val;
    for (int i = 0; i < slp_width; i++) {
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
    for (int i = 0; i < slp_width; i++) {
      ptr += dynamic_cast<LocalLoadStmt *>(building_pack[i])->ptr;
    }
    tmp_stmt = std::make_unique<LocalLoadStmt>(ptr);
    tmp_stmt->ret_type.width = stmt->ret_type.width * slp_width;
    update_type(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    tmp_stmt =
        std::make_unique<LocalStoreStmt>(tmp_operands[0], tmp_operands[1]);
    tmp_stmt->ret_type.width = stmt->ret_type.width * slp_width;
    update_type(stmt);
  }

  void visit(GlobalPtrStmt *stmt) override {
    std::vector<Stmt *> indices = tmp_operands;
    LaneAttribute<SNode *> snodes;
    for (int i = 0; i < slp_width; i++) {
      snodes += building_pack[i]->as<GlobalPtrStmt>()->snode;
    }
    tmp_stmt = Stmt::make<GlobalPtrStmt>(snodes, indices);
    tmp_stmt->ret_type.width = stmt->ret_type.width * slp_width;
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

  void visit(ElementShuffleStmt *stmt) override {
    LaneAttribute<VectorElement> elements;
    for (int i = 0; i < slp_width; i++) {
      elements += building_pack[i]->as<ElementShuffleStmt>()->elements;
    }
    tmp_stmt = Stmt::make<ElementShuffleStmt>(elements);
    update_type(stmt);
  }

  Stmt *find_stmt(const Pack &pack) {
    TC_ASSERT((int)pack.size() == slp_width);
    for (int i = 0; i < (int)existing_stmts.size(); i++) {
      bool match = true;
      for (int j = 0; j < slp_width; j++) {
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
    for (int i = 0; i < slp_width; i++) {
      if (inside.find(pack[i]) == inside.end())
        return nullptr;
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
          auto previous =
              pack[j]->as<LocalLoadStmt>()->previous_store_or_alloca_in_block();
          /*
          if (previous->is<LocalStoreStmt>()) {
            previous = previous->as<LocalStoreStmt>()->data;
          }
          */
          if (previous)
            operand_pack.push_back(previous);
        }
        if (operand_pack.size() != 0) {
          TC_ASSERT((int)operand_pack.size() == slp_width);
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
    for (int i = 0; i < (int)building_pack.size(); i++) {
      (*rec)[building_pack[i]] = std::make_pair(tmp_stmt.get(), i);
    }
    auto ret = new_stmts.push_back(std::move(tmp_stmt));

    int pos = -1;
    for (int i = 0; i < (int)input_statements->size(); i++) {
      if (pack[0] == (*input_statements)[i]) {
        pos = i;
        break;
      }
    }
    TC_ASSERT(pos != -1);
    position[ret] = pos;
    existing_stmts.push_back(std::make_pair(pack, ret));
    return ret;
  }

  void replace(Stmt *old_stmt, Stmt *new_stmt, int offset) {
    for (int i = 0; i < (int)block->statements.size(); i++) {
      auto stmt = block->statements[i].get();
      if (inside.find(stmt) != inside.end())
        continue;  // this is a statement being SLP vectorized..
      for (auto ope : stmt->operands) {
        if (*ope == old_stmt) {
          TC_ASSERT(old_stmt->width() == 1);
          auto shuffle =
              Stmt::make<ElementShuffleStmt>(VectorElement(new_stmt, 0));
          *ope = shuffle.get();
          shuffles.push_back(std::move(shuffle));
        }
      }
    }
  }

  // replace with BBlock with SLP'ed block
  VecStatement run(Block *block,
                   int width,
                   std::vector<Stmt *> &input_statements,
                   PackRecord *rec) {
    this->rec = rec;
    this->block = block;
    this->slp_width = width;
    this->input_statements = &input_statements;
    inside = std::set<Stmt *>(input_statements.begin(), input_statements.end());
    visited.clear();
    auto &stmts = input_statements;
    while (1) {
      TC_INFO("Seeding...");
      // Find the last statement
      Stmt *last_stmt = nullptr;
      for (int i = stmts.size() - 1; i >= 0; i--) {
        if (!stmts[i]->is<PragmaSLPStmt>() &&
            visited.find(stmts[i]) == visited.end()) {
          last_stmt = stmts[i];
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
          seed_statements.push_back(stmts[i]);
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
    for (auto &s : shuffles)
      new_stmts.stmts.push_back(std::move(s));
    return std::move(new_stmts);
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
              for (int j = 0; j < slp_width; j++) {
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
  PackRecord rec;

  SLPVectorize() {
    allow_undefined_visitor = true;
    // invoke_default_visitor = true;
  }

  // A SLP segment is a subarray of block->statements with the same SLP width
  // This method transforms the first SLP segment.
  // After the invocation the block may be invalid. This is can be fixed by
  // inserting ElementShuffleStmt's
  void slp_attempt(Block *block) {
    std::vector<Stmt *> current_segment;

    int first_pragma_slp_location = -1;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      if (block->statements[i]->is<PragmaSLPStmt>()) {
        first_pragma_slp_location = i;
        break;
      }
    }

    if (first_pragma_slp_location == -1)
      return;

    // now, insert shuffles to make sure SLPing the previous segment does not
    // break the block

    int second_pragma_slp_location = -1;
    for (int i = first_pragma_slp_location + 1;
         i < (int)block->statements.size(); i++) {
      if (block->statements[i]->is<PragmaSLPStmt>()) {
        second_pragma_slp_location = i;
        break;
      }
    }

    if (second_pragma_slp_location == -1) {
      // until the end...
      second_pragma_slp_location = (int)block->statements.size();
    }

    std::vector<pStmt> shuffles;
    for (int i = first_pragma_slp_location + 1; i < second_pragma_slp_location;
         i++) {
      auto stmt = block->statements[i].get();
      if (stmt->is<LocalLoadStmt>()) {
        auto s = stmt->as<LocalLoadStmt>();
        for (int l = 0; l < s->width(); l++) {
          auto old_alloca = s->ptr[l].var;
          if (rec.find(old_alloca) != rec.end()) {
            s->ptr[l].var = rec[old_alloca].first;
            s->ptr[l].offset = rec[old_alloca].second;
          }
        }
      } else {
        for (auto ope : stmt->operands) {
          if (rec.find(*ope) != rec.end()) {
            TC_P(stmt->id);
            // TC_P((*ope)->id);
            // TC_ASSERT((*ope)->width() == 1);
            auto shuffle = Stmt::make<ElementShuffleStmt>(
                VectorElement(rec[*ope].first, rec[*ope].second));
            *ope = shuffle.get();
            shuffles.push_back(std::move(shuffle));
          }
        }
      }
    }

    for (int i = 0; i < (int)shuffles.size(); i++) {
      block->insert(std::move(shuffles[i]), first_pragma_slp_location + i);
    }
    second_pragma_slp_location += (int)shuffles.size();

    int current_slp_width = block->statements[first_pragma_slp_location]
                                ->as<PragmaSLPStmt>()
                                ->slp_width;

    std::vector<Stmt *> vec;
    for (int i = first_pragma_slp_location + 1; i < second_pragma_slp_location;
         i++) {
      vec.push_back(block->statements[i].get());
    }
    TC_INFO("Before SLP");
    irpass::print(context->root());
    auto slp = BasicBlockSLP();
    block->replace_statements_in_range(
        first_pragma_slp_location, second_pragma_slp_location,
        slp.run(block, current_slp_width, vec, &rec));
    TC_P(first_pragma_slp_location);
    TC_P(second_pragma_slp_location);
    TC_INFO("SLPed...");
    irpass::print(context->root());
    throw IRModifiedException();
  }

  void fix_indirect_alloca_ref(Block *block) {
    for (auto &stmt_ : block->statements) {
      if (stmt_->is<LocalLoadStmt>()) {
        auto stmt = stmt_->as<LocalLoadStmt>();
        for (int l = 0; l < stmt->width(); l++) {
          if (stmt->ptr[l].var->is<ElementShuffleStmt>()) {
            int offset = stmt->ptr[l].offset;
            auto shuffle = stmt->ptr[l].var->as<ElementShuffleStmt>();
            stmt->ptr[l].var = shuffle->elements[offset].stmt;
            offset = shuffle->elements[offset].index;
            stmt->ptr[l].offset = offset;
          }
        }
      }
    }
  }

  void visit(Block *block) {
    TC_ASSERT(block->statements.size() != 0);
    while (true) {
      try {
        slp_attempt(block);
      } catch (IRModifiedException) {
        continue;
      }
      break;  // if no IRModifiedException
    }
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
    fix_indirect_alloca_ref(block);
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
