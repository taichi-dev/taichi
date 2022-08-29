#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/check_out_of_bound.h"
#include "taichi/transforms/utils.h"
#include <set>

TLANG_NAMESPACE_BEGIN

// TODO: also check RangeAssumptionStmt

class CheckOutOfBound : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::set<int> visited;
  DelayedIRModifier modifier;
  std::string kernel_name;

  explicit CheckOutOfBound(const std::string &kernel_name)
      : BasicStmtVisitor(), visited(), kernel_name(kernel_name) {
  }

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->ptr != nullptr) {
      TI_ASSERT(stmt->ptr->is<GlobalPtrStmt>());
      // We have already done the check on its ptr argument. No need to do
      // anything here.
      return;
    }

    // TODO: implement bound check here for other situations.
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (is_done(stmt))
      return;
    auto snode = stmt->snode;
    bool has_offset = !(snode->index_offsets.empty());
    auto new_stmts = VecStatement();
    auto zero = new_stmts.push_back<ConstStmt>(TypedConstant(0));
    Stmt *result = new_stmts.push_back<ConstStmt>(TypedConstant(true));

    std::string msg =
        fmt::format("(kernel={}) Accessing field ({}) of size (", kernel_name,
                    snode->get_node_type_name_hinted());
    std::string offset_msg = "offset (";
    std::vector<Stmt *> args;
    for (int i = 0; i < stmt->indices.size(); i++) {
      int offset_i = has_offset ? snode->index_offsets[i] : 0;

      // Note that during lower_ast, index arguments to GlobalPtrStmt are
      // already converted to [0, +inf) range.

      auto lower_bound = zero;
      auto check_lower_bound = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::cmp_ge, stmt->indices[i], lower_bound);
      int size_i = snode->shape_along_axis(i);
      int upper_bound_i = size_i;
      auto upper_bound =
          new_stmts.push_back<ConstStmt>(TypedConstant(upper_bound_i));
      auto check_upper_bound = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::cmp_lt, stmt->indices[i], upper_bound);
      auto check_i = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::bit_and, check_lower_bound, check_upper_bound);
      result = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, result,
                                                 check_i);
      if (i > 0) {
        msg += ", ";
        offset_msg += ", ";
      }
      msg += std::to_string(size_i);
      offset_msg += std::to_string(offset_i);

      auto input_index = stmt->indices[i];
      if (offset_i != 0) {
        auto offset = new_stmts.push_back<ConstStmt>(TypedConstant(offset_i));
        input_index = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::add,
                                                        input_index, offset);
      }
      args.emplace_back(input_index);
    }
    offset_msg += ") ";
    msg += ") " + (has_offset ? offset_msg : "") + "with indices (";
    for (int i = 0; i < stmt->indices.size(); i++) {
      if (i > 0)
        msg += ", ";
      msg += "%d";
    }
    msg += ")";
    msg += "\n" + stmt->tb;

    new_stmts.push_back<AssertStmt>(result, msg, args);
    modifier.insert_before(stmt, std::move(new_stmts));
    set_done(stmt);
  }

  void visit(BinaryOpStmt *stmt) override {
    // Insert assertions if debug is on
    if (is_done(stmt)) {
      return;
    }
    if (stmt->op_type == BinaryOpType::pow) {
      if (is_integral(stmt->rhs->ret_type) &&
          is_integral(stmt->lhs->ret_type)) {
        auto compare_rhs = Stmt::make<ConstStmt>(TypedConstant(0));
        auto compare = std::make_unique<BinaryOpStmt>(
            BinaryOpType::cmp_ge, stmt->rhs, compare_rhs.get());
        compare->ret_type = PrimitiveType::i32;
        std::string msg = "Negative exponent for integer pows are not allowed";
        msg += "\n" + stmt->tb;
        auto assert_stmt = std::make_unique<AssertStmt>(compare.get(), msg,
                                                        std::vector<Stmt *>());
        assert_stmt->accept(this);
        modifier.insert_before(stmt, std::move(compare_rhs));
        modifier.insert_before(stmt, std::move(compare));
        modifier.insert_before(stmt, std::move(assert_stmt));
        set_done(stmt);
      }
    }
  }

  static bool run(IRNode *node,
                  const CompileConfig &config,
                  const std::string &kernel_name) {
    CheckOutOfBound checker(kernel_name);
    bool modified = false;
    while (true) {
      node->accept(&checker);
      if (checker.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }
    if (modified)
      irpass::type_check(node, config);
    return modified;
  }
};

const PassID CheckOutOfBoundPass::id = "CheckOutOfBoundPass";

namespace irpass {

bool check_out_of_bound(IRNode *root,
                        const CompileConfig &config,
                        const CheckOutOfBoundPass::Args &args) {
  TI_AUTO_PROF;
  return CheckOutOfBound::run(root, config, args.kernel_name);
}

}  // namespace irpass

TLANG_NAMESPACE_END
