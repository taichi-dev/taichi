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

  struct JITEvaluatorId
  {
    int op;
    DataType ret, lhs, rhs;
    bool is_binary;
    bool cast_by_value;

    explicit operator int() const // make STL map happy
    {
      return (int)op | (int)!!is_binary << 7 | (int)!!cast_by_value << 6
        | (int) ret << 8 | (int) lhs << 16 | (int) rhs << 24;
    }

    UnaryOpType unary_op() const
    {
      return (UnaryOpType) op;
    }

    BinaryOpType binary_op() const
    {
      return (BinaryOpType) op;
    }

#if 0
    bool operator==(const JITEvaluatorId &b)
    {
      return op == b.op && ret == b.ret && lhs == b.lhs && rhs == b.rhs;
    }

    struct hash {
      size_t operator()(const JITEvaluatorId &id)
      {
        hash<int> hop;
        hash<DataType> hdt;
        return hop(op | is_binary() << 7)
          ^ hdt(ret) ^ hdt(lhs) ^ hdt(rhs);
      }
    };
#endif
  };

  static Kernel *get_jit_evaluator_kernel(JITEvaluatorId const &id)
  {
    auto &cache = get_current_program().jit_evaluator_cache;
    int iid = int(id);
    auto it = cache.find(iid);
    if (it != cache.end()) // cached?
      return it->second.get();
    static int jic = 0; // X: race?
    auto kernel_name = fmt::format("jit_evaluator_{}", jic++);
    auto func = [&] () {
      auto lhstmt = Stmt::make<ArgLoadStmt>(1, false);
      auto rhstmt = Stmt::make<ArgLoadStmt>(2, false);
      pStmt oper;
      if (id.is_binary) {
        oper = Stmt::make<BinaryOpStmt>(id.binary_op(), lhstmt.get(), rhstmt.get());
      } else {
        oper = Stmt::make<UnaryOpStmt>(id.unary_op(), lhstmt.get());
        if (id.unary_op() == UnaryOpType::cast) {
          auto ustmt = oper->cast<UnaryOpStmt>();
          ustmt->cast_type = id.rhs;
          ustmt->cast_by_value = id.cast_by_value;
        }
      }
      auto ret = Stmt::make<ArgStoreStmt>(0, oper.get());
      current_ast_builder().insert(std::move(lhstmt));
      if (id.is_binary)
        current_ast_builder().insert(std::move(rhstmt));
      current_ast_builder().insert(std::move(oper));
      current_ast_builder().insert(std::move(ret));
    };
    auto ker = std::make_unique<Kernel>(get_current_program(), func, kernel_name);
    ker->insert_arg(id.ret, false);
    ker->insert_arg(id.lhs, false);
    if (id.is_binary)
      ker->insert_arg(id.rhs, false);
    ker->mark_arg_return_value(0, true);
    auto *ker_ptr = ker.get();
    TI_TRACE("Saving JIT evaluator cache entry id={}", iid);
    cache[iid] = std::move(ker);
    return ker_ptr;
  }

  static bool jit_from_binary_op(TypedConstant &ret, BinaryOpStmt *stmt,
      const TypedConstant &lhs, const TypedConstant &rhs)
  {
    JITEvaluatorId id{(int)stmt->op_type, ret.dt, lhs.dt, rhs.dt,
      true, false};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().get_context();
    //TI_INFO("JITARGSf = {} {}", lhs.val_f32, rhs.val_f32);
    TI_INFO("JITARGSi = {} {}", lhs.val_i32, rhs.val_i32);
    ctx.set_arg<int64_t>(0, 233);
    ctx.set_arg<int64_t>(1, lhs.val_i64);
    ctx.set_arg<int64_t>(2, rhs.val_i64);
    irpass::print(ker->ir);
    (*ker)();
    ret.val_i64 = ctx.get_arg<int64_t>(0);
    //TI_INFO("JITEVALf = {}", ret.val_f32);
    TI_INFO("JITEVALi = {}", ret.val_i32);
    return true;
  }

  static bool jit_from_unary_op(TypedConstant &ret, UnaryOpStmt *stmt,
      const TypedConstant &lhs)
  {
    JITEvaluatorId id{(int)stmt->op_type, ret.dt, lhs.dt, stmt->cast_type,
      false, stmt->cast_by_value};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().get_context();
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
    if (jit_from_binary_op(new_constant, stmt, lhs->val[0], rhs->val[0])) {
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
    if (jit_from_unary_op(new_constant, stmt, lhs->val[0])) {
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
