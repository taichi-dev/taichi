#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

namespace {

using namespace taichi;
using namespace taichi::lang;

class CreateBitStructStores : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  CreateBitStructStores() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  static void run(IRNode *root) {
    CreateBitStructStores pass;
    root->accept(&pass);
  }

  void visit(GlobalStoreStmt *stmt) {
    auto get_ch = stmt->ptr->cast<GetChStmt>();
    if (!get_ch || get_ch->input_snode->type != SNodeType::bit_struct)
      return;

    // We only handle bit_struct pointers here. The currently supported data
    // types are CustomIntType and CustomFloatType without exponents.
    auto dtype = get_ch->output_snode->dt;
    if (dtype->is<CustomIntType>() ||
        dtype->as<CustomFloatType>()->get_exponent_type() == nullptr) {
      auto s = Stmt::make<BitStructStoreStmt>(get_ch->input_ptr,
                                              std::vector<int>{get_ch->chid},
                                              std::vector<Stmt *>{stmt->data});
      stmt->replace_with(VecStatement(std::move(s)));
    }
  }
};

class MergeBitStructStores : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  MergeBitStructStores() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  static void run(IRNode *root) {
    while (true) {
      MergeBitStructStores pass;
      root->accept(&pass);
      if (!pass.modified_)
        break;
    }
  }

  void visit(Block *block) override {
    auto &statements = block->statements;
    std::unordered_map<Stmt *, std::vector<BitStructStoreStmt *>>
        ptr_to_bit_struct_stores;
    std::vector<Stmt *> statements_to_delete;
    for (int i = 0; i <= (int)statements.size(); i++) {
      // TODO: in some cases BitSturctStoreStmts across container statements can
      // still be merged, similar to basic block v.s. CFG optimizations.
      if (i == statements.size() || statements[i]->is_container_statement()) {
        for (const auto &item : ptr_to_bit_struct_stores) {
          auto ptr = item.first;
          auto stores = item.second;
          if (stores.size() == 1)
            continue;
          std::map<int, Stmt *> values;
          for (auto s : stores) {
            for (int j = 0; j < (int)s->ch_ids.size(); j++) {
              values[s->ch_ids[j]] = s->values[j];
            }
          }
          std::vector<int> ch_ids;
          std::vector<Stmt *> store_values;
          for (auto &ch_id_and_value : values) {
            ch_ids.push_back(ch_id_and_value.first);
            store_values.push_back(ch_id_and_value.second);
          }
          // Now erase all (except the last) related BitSturctStoreStmts.
          // Replace the last one with a merged version.
          for (int j = 0; j < (int)stores.size() - 1; j++) {
            statements_to_delete.push_back(stores[j]);
          }
          stores.back()->replace_with(
              Stmt::make<BitStructStoreStmt>(ptr, ch_ids, store_values));
          modified_ = true;
        }
        ptr_to_bit_struct_stores.clear();
        continue;
      }
      if (auto stmt = statements[i]->cast<BitStructStoreStmt>()) {
        ptr_to_bit_struct_stores[stmt->ptr].push_back(stmt);
      }
    }

    for (auto stmt : statements_to_delete) {
      block->erase(stmt);
    }

    for (auto &stmt : statements) {
      stmt->accept(this);
    }
  }

 private:
  bool modified_{false};
};

}  // namespace

TLANG_NAMESPACE_BEGIN

namespace irpass {
void optimize_bit_struct_stores(IRNode *root) {
  TI_AUTO_PROF;
  CreateBitStructStores::run(root);
  die(root);  // remove unused GetCh
  MergeBitStructStores::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
