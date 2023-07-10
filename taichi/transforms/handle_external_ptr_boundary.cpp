#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/check_out_of_bound.h"
#include "taichi/transforms/utils.h"
#include <set>

namespace taichi::lang {

class HandleExternalPtrBound : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::set<int> visited;
  DelayedIRModifier modifier;

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void visit(ExternalPtrStmt *stmt) override {
    if (stmt->boundary == BoundaryMode::kUnsafe || is_done(stmt))
      return;
    if (stmt->boundary == BoundaryMode::kClamp) {
      auto new_stmts = VecStatement();
      auto zero = new_stmts.push_back<ConstStmt>(TypedConstant(0));
      for (int i = 0; i < stmt->indices.size(); i++) {
        auto lower_bound = zero;
        auto checked_index = new_stmts.push_back<BinaryOpStmt>(
            BinaryOpType::max, stmt->indices[i], lower_bound);
        auto upper_bound =
            new_stmts.push_back<ExternalTensorShapeAlongAxisStmt>(
                /*axis=*/i,
                /*arg_id=*/stmt->base_ptr->as<ArgLoadStmt>()->arg_id);
        auto one = new_stmts.push_back<ConstStmt>(TypedConstant(1));
        auto valid_upper = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::sub,
                                                             upper_bound, one);

        checked_index = new_stmts.push_back<BinaryOpStmt>(
            BinaryOpType::min, checked_index, valid_upper);
        stmt->indices[i] = checked_index;
      }

      modifier.insert_before(stmt, std::move(new_stmts));
      set_done(stmt);
    }
  }

  // TODO: As offset information per dimension is lacking, only the accumulated
  // index is checked.
  void visit(MatrixPtrStmt *stmt) override {
    if (is_done(stmt) || !stmt->offset_used_as_index())
      return;

    if (stmt->origin->is<ExternalPtrStmt>() &&
        stmt->origin->as<ExternalPtrStmt>()->boundary == BoundaryMode::kClamp) {
      auto const &matrix_shape = stmt->get_origin_shape();
      int max_valid_index = 1;
      for (int i = 0; i < matrix_shape.size(); i++) {
        max_valid_index *= matrix_shape[i];
      }
      // index starts from 0, max_valid_index = size(matrix) - 1
      max_valid_index -= 1;

      auto index = stmt->offset;
      auto new_stmts = VecStatement();
      auto zero = new_stmts.push_back<ConstStmt>(TypedConstant(0));

      auto lower_bound = zero;

      auto checked_index = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::max, index, lower_bound);
      auto upper_bound =
          new_stmts.push_back<ConstStmt>(TypedConstant(max_valid_index));
      checked_index = new_stmts.push_back<BinaryOpStmt>(
          BinaryOpType::min, checked_index, upper_bound);

      stmt->offset = checked_index;
      modifier.insert_before(stmt, std::move(new_stmts));
      set_done(stmt);
    }
  }

  static bool run(IRNode *node, const CompileConfig &config) {
    HandleExternalPtrBound checker;
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

namespace irpass {

void handle_external_ptr_boundary(IRNode *root, const CompileConfig &config) {
  TI_AUTO_PROF;
  HandleExternalPtrBound::run(root, config);
}

}  // namespace irpass

}  // namespace taichi::lang
