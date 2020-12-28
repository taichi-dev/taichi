#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

namespace {

using namespace taichi;
using namespace taichi::lang;

class CreateBitStructStores : public BasicStmtVisitor {
 public:
  CreateBitStructStores() {
    allow_undefined_visitor = true;
    invoke_default_visitor = false;
  }

  void run(IRNode *root) {
    root->accept(this);
  }

  void visit(GlobalStoreStmt *stmt) {
    auto get_ch = stmt->ptr->cast<GetChStmt>();
    if (!get_ch || get_ch->input_snode->type != SNodeType::bit_struct)
      return;

    // We only handle bit_struct pointers here
    auto s = Stmt::make<BitStructStoreStmt>(get_ch->input_ptr,
                                            std::vector<int>{get_ch->chid},
                                            std::vector<Stmt *>{stmt->data});
    stmt->replace_with(VecStatement(std::move(s)));
  }
};
}  // namespace

TLANG_NAMESPACE_BEGIN

namespace irpass {
void optimize_bit_struct_stores(IRNode *root) {
  TI_AUTO_PROF;
  CreateBitStructStores opt;
  opt.run(root);
  die(root);  // remove unused GetCh
}

}  // namespace irpass

TLANG_NAMESPACE_END
