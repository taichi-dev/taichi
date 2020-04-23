#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include <deque>
#include <set>
#include <cmath>

TLANG_NAMESPACE_BEGIN

bool TypedConstant::from_unary_op(UnaryOpType op, const TypedConstant &rhs)
{
#define PER_OP(op, o) \
    case UnaryOpType::op: this->val_i32 = o(rhs.val_i32); break;
  switch (op) {
  PER_OP(neg, -)
  PER_OP(sqrt, std::sqrt)
  PER_OP(log, std::log)
  PER_OP(exp, std::exp)
  PER_OP(abs, std::abs)
  PER_OP(sin, std::sin)
  PER_OP(cos, std::cos)
  PER_OP(tan, std::tan)
  PER_OP(asin, std::asin)
  PER_OP(acos, std::acos)
  PER_OP(tanh, std::tanh)
  PER_OP(rsqrt, 1 / std::sqrt)
  PER_OP(floor, std::floor)
  PER_OP(ceil, std::ceil)
  PER_OP(bit_not, ~)
  PER_OP(logic_not, !)
  PER_OP(inv, 1 /)
  default: return false;
  }
#undef PER_OP
  return true;
}

bool TypedConstant::from_binary_op(BinaryOpType op,
    const TypedConstant &lhs, const TypedConstant &rhs)
{
#define PER_OP(op, o) \
    case BinaryOpType::op: this->val_i32 = lhs.val_i32 o rhs.val_i32; break;
#define PER_OF(op, f) \
    case BinaryOpType::op: this->val_i32 = f(lhs.val_i32, rhs.val_i32); break;
  switch (op) {
  PER_OP(add, +)
  PER_OP(sub, -)
  PER_OP(mul, *)
  PER_OP(div, /)
  PER_OP(truediv, /) // XXX: is this same as div?
  PER_OP(floordiv, /) // XXX: do we have std::floordiv?
  // PER_OP(mod, %) // XXX: raw_mod or python-mod?
  PER_OP(bit_or, |)
  PER_OP(bit_and, &)
  PER_OP(bit_xor, ^)
  PER_OP(cmp_lt, <)
  PER_OP(cmp_le, <=)
  PER_OP(cmp_gt, >)
  PER_OP(cmp_ge, >=)
  PER_OP(cmp_eq, ==)
  PER_OP(cmp_ne, !=)
  PER_OF(max, std::max)
  PER_OF(min, std::min)
  PER_OF(pow, std::pow)
  PER_OF(atan2, std::atan2)
  default: return false;
  }
#undef PER_OP
  return true;
}

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ConstantFold() : BasicStmtVisitor() {
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->width() == 1 && stmt->op_type == UnaryOpType::cast &&
        stmt->cast_by_value && stmt->operand->is<ConstStmt>()) {
      auto input = stmt->operand->as<ConstStmt>()->val[0];
      auto src_type = stmt->operand->ret_type.data_type;
      auto dst_type = stmt->ret_type.data_type;
      TypedConstant new_constant(dst_type);
      bool success = false;
      if (src_type == DataType::f32) {
        auto v = input.val_float32();
        if (dst_type == DataType::i32) {
          new_constant.val_i32 = int32(v);
          success = true;
        }
      } else if (src_type == DataType::i32) {
        auto v = input.val_int32();
        if (dst_type == DataType::f32) {
          new_constant.val_f32 = float32(v);
          success = true;
        }
      }

      if (success) {
        auto evaluated =
            Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
        stmt->replace_with(evaluated.get());
        stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs || !rhs)
      return;
    if (stmt->width() != 1 || stmt->ret_type.data_type != DataType::i32)
      return;
    auto dst_type = DataType::i32;
    TypedConstant new_constant(dst_type);
    if (new_constant.from_binary_op(stmt->op_type, lhs->val[0], rhs->val[0])) {
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

class ConstantFoldJIT : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::vector<BinaryOpStmt *> bops;
  std::vector<UnaryOpStmt *> uops;

  ConstantFoldJIT() : BasicStmtVisitor() {
  }

  void visit(UnaryOpStmt *stmt) override {
    uops.push_back(stmt);
  }

  void visit(BinaryOpStmt *stmt) override {
    bops.push_back(stmt);
  }

  static Kernel *get_jit_constexpr_kernel(Stmt *stmt) {
    // BEGIN: generic visitor to extract all oprand
    auto bop = stmt->cast<BinaryOpStmt>();
    auto lhs = bop->lhs;
    auto rhs = bop->rhs;
    // END: generic visitor to extract all oprand
    auto kernel_name = fmt::format("jit_constexpr_{}", 0);
    auto func = [] () {
    };
    auto ker = new Kernel(get_current_program(), func, kernel_name); // ???
    // ker->ir = insert(bop, lhs, rhs)!!!!
    ker->set_arch(Arch::x64); // X: host_arch
    // ker->is_accessor = true; // X: is_tiny_kernel?
    return ker;
  }

  static void launch_jit(Stmt *stmt) {
    Kernel *jit_kernel = get_jit_constexpr_kernel(stmt);
    get_current_program().synchronize();
    get_current_program().config.no_cp2o = true;
    TI_INFO("Launching JIT evaluator IN");
    (*jit_kernel)();
    TI_INFO("Launching JIT evaluator OUT");
    get_current_program().config.no_cp2o = false;
  }

  static void run(IRNode *node) {
    TI_INFO("!!! FANLE");
    ConstantFoldJIT folder;
    node->accept(&folder);
    if (folder.bops.size()) {
      auto back = folder.bops.back();
      folder.bops.pop_back();
      launch_jit(back);
    }
  }
};

namespace irpass {

void constant_fold(IRNode *root) {
  return ConstantFoldJIT::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
