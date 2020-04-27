#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/ir/snode.h"
#include <deque>
#include <set>
#include <cmath>

TLANG_NAMESPACE_BEGIN

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ConstantFold() : BasicStmtVisitor() {
  }

#define INVALID_TYPE ((DataType)0xff)

  struct BinaryEvaluatorId
  {
    int op;
    DataType ret, lhs, rhs;

    bool is_binary() const
    {
      return rhs != INVALID_TYPE;
    }

    explicit operator int() const // make STL map happy
    {
      return (int)op | (int)is_binary() << 7
        | (int) ret << 8 | (int) lhs << 16 | (int) rhs << 24;
    }

#if 0
    struct hash {
      size_t operator()(const BinaryEvaluatorId &id)
      {
        hash<int> hop;
        hash<DataType> hdt;
        return hop(op | is_binary() << 7)
          ^ hdt(ret) ^ hdt(lhs) ^ hdt(rhs);
      }
    };
#endif
  };

  static Kernel *get_jit_evaluator_kernel(BinaryEvaluatorId const &id)
  {
    auto &cache = get_current_program().jit_evaluator_cache;
    int iid = int(id);
    TI_INFO("IN {}", iid);
    auto it = cache.find(iid);
    TI_INFO("OUT");
    if (it != cache.end()) // cached?
      return it->second.get();
    static int jic = 0; // X: race?
    auto kernel_name = fmt::format("jit_evaluator_{}", jic++);
    auto func = [&] () {
      auto lhstmt = Stmt::make<ArgLoadStmt>(1, false);
      auto rhstmt = Stmt::make<ArgLoadStmt>(2, false);
      pStmt oper;
      if (id.is_binary())
        oper = Stmt::make<BinaryOpStmt>((BinaryOpType)id.op, lhstmt.get(), rhstmt.get());
      else
        oper = Stmt::make<UnaryOpStmt>((UnaryOpType)id.op, lhstmt.get());
      auto ret = Stmt::make<ArgStoreStmt>(0, oper.get());
      current_ast_builder().insert(std::move(lhstmt));
      if (id.is_binary())
        current_ast_builder().insert(std::move(rhstmt));
      current_ast_builder().insert(std::move(oper));
      current_ast_builder().insert(std::move(ret));
    };
    auto ker = std::make_unique<Kernel>(get_current_program(), func, kernel_name);
    ker->insert_arg(id.ret, false);
    ker->insert_arg(id.lhs, false);
    if (id.is_binary())
      ker->insert_arg(id.rhs, false);
    ker->mark_arg_return_value(0, true);
    auto *ker_ptr = ker.get();
    TI_INFO("SAV {}", iid);
    cache[iid] = std::move(ker);
    return ker_ptr;
  }

  static bool jit_from_binary_op(TypedConstant &ret, BinaryOpType op,
      const TypedConstant &lhs, const TypedConstant &rhs)
  {
    BinaryEvaluatorId id{(int)op, ret.dt, lhs.dt, rhs.dt};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().context;
    ctx.set_arg<int64_t>(1, lhs.val_i64);
    ctx.set_arg<int64_t>(2, rhs.val_i64);
    irpass::print(ker->ir);
    (*ker)();
    ret.val_i64 = ctx.get_arg<int64_t>(0);
    return true;
  }

  static bool jit_from_unary_op(TypedConstant &ret, UnaryOpType op,
      const TypedConstant &lhs)
  {
    BinaryEvaluatorId id{(int)op, ret.dt, lhs.dt, INVALID_TYPE};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().context;
    ctx.set_arg<int64_t>(1, lhs.val_i64);
    irpass::print(ker->ir);
    (*ker)();
    ret.val_i64 = ctx.get_arg<int64_t>(0);
    return true;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs || !rhs)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type.data_type;
    TypedConstant new_constant(dst_type);
    if (jit_from_binary_op(new_constant, stmt->op_type, lhs->val[0], rhs->val[0])) {
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    auto lhs = stmt->operand->cast<ConstStmt>();
    if (!lhs)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type.data_type;
    TypedConstant new_constant(dst_type);
    if (jit_from_unary_op(new_constant, stmt->op_type, lhs->val[0])) {
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  static void run(IRNode *node) {
    ConstantFold folder;
    while (true) {
      bool modified = false;
      try {
        node->accept(&folder);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void constant_fold(IRNode *root) {
  return ConstantFold::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
