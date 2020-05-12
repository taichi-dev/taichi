// Superword level vectorization

#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include <typeinfo>
#include <set>

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
  void update_type(Stmt *stmt) {
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
      snodes += building_pack[i]->as<GlobalPtrStmt>()->snodes;
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

  void visit(UnaryOpStmt *stmt) override {
    tmp_stmt = std::make_unique<UnaryOpStmt>(
        dynamic_cast<UnaryOpStmt *>(building_pack[0])->op_type,
        tmp_operands[0]);
    tmp_stmt->as<UnaryOpStmt>()->cast_type = stmt->cast_type;
    update_type(stmt);
    /*
    if (tmp_stmt->as<UnaryOpStmt>()->op_type == UnaryOpType::cast) {
      tmp_stmt->element_type() = tmp_stmt->as<UnaryOpStmt>()->cast_type;
    }
    */
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
    for (int i = 0; i < (int)pack.size(); i++) {
      if (rec->find(pack[i]) != rec->end()) {
        return rec->find(pack[i])->second.first;
      }
    }
    TI_ASSERT((int)pack.size() == slp_width);
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
    bool identical = true;
    for (int i = 0; i < slp_width; i++) {
      if (pack[i] != pack[0]) {
        identical = false;
      }
    }
    if (slp_width > 1 && identical) {
      /*
      if (visited.find(pack[0]) == visited.end())
        visited.insert(pack[0]);
      */
      return pack[0];
    }
    for (int i = 0; i < slp_width; i++) {
      if (inside.find(pack[i]) == inside.end()) {
        return pack[i];
      }
      // fmt::print(" {} ", pack[i]->id);
      TI_ASSERT(visited.find(pack[i]) == visited.end());
      visited.insert(pack[i]);
    }
    // fmt::print("\n");
    Pack operands;
    if (!pack[0]->is<LocalLoadStmt>()) {
      for (int i = 0; i < pack[0]->num_operands(); i++) {
        Pack operand_pack;
        for (int j = 0; j < (int)pack.size(); j++) {
          operand_pack.push_back(pack[j]->operand(i));
        }
        operands.push_back(build(operand_pack));
      }
    } else {
      TI_ASSERT(pack[0]->width() == 1);
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
          TI_ASSERT((int)operand_pack.size() == slp_width);
          operands.push_back(build(operand_pack));
        }
      }
    }
    tmp_operands = operands;
    building_pack = pack;
    TI_ASSERT(tmp_stmt == nullptr);
    pack[0]->accept(this);
    TI_ASSERT(tmp_stmt != nullptr);
    tmp_operands.clear();
    for (int i = 0; i < (int)building_pack.size(); i++) {
      TI_ASSERT(rec->find(building_pack[i]) == rec->end());
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
    TI_ASSERT(pos != -1);
    position[ret] = pos;
    existing_stmts.push_back(std::make_pair(pack, ret));
    /*
    for (int i = 0; i < slp_width; i++) {
      fmt::print(" {} ", pack[i]->id);
    }
    fmt::print(" -> {} ", ret->id);
    fmt::print("\n");
    */
    return ret;
  }

  void replace(Stmt *old_stmt, Stmt *new_stmt, int offset){
      TI_NOT_IMPLEMENTED
      /*
      for (int i = 0; i < (int)block->statements.size(); i++) {
        auto stmt = block->statements[i].get();
        if (inside.find(stmt) != inside.end())
          continue;  // this is a statement being SLP vectorized..
        for (auto ope : stmt->operands) {
          if (*ope == old_stmt) {
            TI_ASSERT(old_stmt->width() == 1);
            auto shuffle =
                Stmt::make<ElementShuffleStmt>(VectorElement(new_stmt, 0));
            *ope = shuffle.get();
            shuffles.push_back(std::move(shuffle));
          }
        }
      }
      */
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
    Stmt *last_last_stmt = nullptr;
    while (1) {
      // TI_INFO("Seeding...");
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
      if (last_stmt == last_last_stmt) {
        TI_ERROR("Last stmt duplicated. Loop detected.");
      }
      last_last_stmt = last_stmt;

      std::vector<Stmt *> seed_statements;

      // from the back, find the other (width - 1) statements of the same type
      for (int i = (int)stmts.size() - 1; i >= 0; i--) {
        auto s = stmts[i];
        if (visited.find(s) == visited.end() &&
            typeid(*last_stmt) == typeid(*s)) {
          // found a stmt of the same type.
          seed_statements.push_back(stmts[i]);
        }
        if ((int)seed_statements.size() == width) {
          break;
        }
      }

      if ((int)seed_statements.size() != width) {
        TI_ERROR("Cannot find enough {} seed statements to start SLP search.",
                 width);
      }
      std::reverse(seed_statements.begin(), seed_statements.end());
      // TI_P(last_stmt->id);
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
            TI_ASSERT(replaced);
          }
        }
      } else if (stmt_->is<LocalStoreStmt>()) {
        auto stmt = stmt_->as<LocalStoreStmt>();
        auto old_stmt = stmt->ptr;
        if (visited.find(old_stmt) != visited.end()) {
          bool replaced = false;
          // replace with packed alloca...
          for (auto &rec : existing_stmts) {
            for (int j = 0; j < slp_width; j++) {
              if (rec.first[j] == old_stmt) {
                TI_ASSERT(j == 0);
                // replace alloca
                stmt->ptr = rec.second;
                replaced = true;
                TI_WARN("Replacing alloca in store");
                break;
              }
            }
            if (replaced)
              break;
          }
          TI_ASSERT(replaced);
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
  // After the invocation the block may be invalid. This is can be dense by
  // inserting ElementShuffleStmt's
  void slp_attempt(Block *block, int iter) {
    std::vector<Stmt *> current_segment;

    int first_pragma_slp_location = -1;

    for (int i = 0; i < (int)block->statements.size(); i++) {
      if (block->statements[i]->is<PragmaSLPStmt>()) {
        first_pragma_slp_location = i;
        break;
      }
    }

    if (first_pragma_slp_location == -1)  // no SLP pragma
      return;

    if (iter == 0 && first_pragma_slp_location != -1) {
      // insert an pragma SLP(1) here to make sure every statement is SLPed.
      block->insert(Stmt::make<PragmaSLPStmt>(1), 0);
      first_pragma_slp_location = 0;
    }

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
    // TI_P(block->statements[first_pragma_slp_location]->id);
    TI_ASSERT(
        block->statements[first_pragma_slp_location]->is<PragmaSLPStmt>());

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
      } else if (stmt->is<ElementShuffleStmt>()) {
        auto s = stmt->as<ElementShuffleStmt>();
        for (int l = 0; l < stmt->width(); l++) {
          auto old_stmt = s->elements[l].stmt;
          TI_ASSERT(s->elements[l].index == 0);
          if (rec.find(old_stmt) != rec.end()) {
            s->elements[l].stmt = rec[old_stmt].first;
            s->elements[l].index = rec[old_stmt].second;
          }
        }
      } else {
        for (int i = 0; i < stmt->num_operands(); i++) {
          auto ope = stmt->operand(i);
          if (rec.find(ope) != rec.end()) {
            auto shuffle = Stmt::make<ElementShuffleStmt>(
                VectorElement(rec[ope].first, rec[ope].second));
            /*
            TI_INFO("Shuffle {}: replaced {} with {}", shuffle->id, ope->id,
                    rec[ope].first->id);
                    */
            stmt->set_operand(i, shuffle.get());
            shuffles.push_back(std::move(shuffle));
          }
        }
      }
    }

    for (int i = 0; i < (int)shuffles.size(); i++) {
      // TI_P(shuffles[i]->id);
      block->insert(std::move(shuffles[i]), first_pragma_slp_location + i + 1);
    }
    second_pragma_slp_location += (int)shuffles.size();

    // TI_P(block->statements[first_pragma_slp_location]->id);
    TI_ASSERT(
        block->statements[first_pragma_slp_location]->is<PragmaSLPStmt>());
    // irpass::print(context->root());
    // TI_P(block->statements[first_pragma_slp_location]->id);
    int current_slp_width = block->statements[first_pragma_slp_location]
                                ->as<PragmaSLPStmt>()
                                ->slp_width;

    std::vector<Stmt *> vec;
    for (int i = first_pragma_slp_location + 1; i < second_pragma_slp_location;
         i++) {
      vec.push_back(block->statements[i].get());
    }
    // TI_INFO("Before SLP");
    auto slp = BasicBlockSLP();
    block->replace_statements_in_range(
        first_pragma_slp_location, second_pragma_slp_location,
        slp.run(block, current_slp_width, vec, &rec));
    /*
    TI_P(first_pragma_slp_location);
    TI_P(second_pragma_slp_location);
    TI_INFO("SLPed...");
    */
    throw IRModified();
  }

  void eliminate_redundant_shuffles(Block *block) {
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
      } else if (stmt_->is<LocalStoreStmt>()) {
        auto stmt = stmt_->as<LocalStoreStmt>();
        if (stmt->ptr->is<ElementShuffleStmt>()) {
          auto ptr = stmt->ptr->as<ElementShuffleStmt>();
          // Assume the shuffle is a trivial copy of vectors
          // TODO: more general case.
          bool trivial = true;
          for (int l = 0; l < ptr->width(); l++) {
            if (ptr->elements[l].index != l) {
              trivial = false;
            }
            if (ptr->elements[l].stmt != ptr->elements[0].stmt) {
              trivial = false;
            }
          }
          if (trivial) {
            stmt->ptr = ptr->elements[0].stmt;
          } else {
            TI_P(stmt->id);
            TI_ERROR(
                "Local store with non trivial shuffling is not yet handled.");
          }
        }
      }
    }
  }

  void visit(Block *block) override {
    if (block->statements.size() == 0) {
      return;
    }
    for (int iter = 0;; iter++) {
      try {
        slp_attempt(block, iter);
      } catch (IRModified) {
        continue;
      }
      break;  // if no IRModified
    }
    for (auto &stmt : block->statements) {
      stmt->accept(this);
    }
    eliminate_redundant_shuffles(block);
  }

  void visit(IfStmt *if_stmt) override {
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
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
