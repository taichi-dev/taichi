#include <cmath>
#include <deque>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "taichi/ir/ir.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/constant_fold.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier;
  Program *program;

  explicit ConstantFold(Program *program)
      : BasicStmtVisitor(), program(program) {
  }

  struct deferred_jit_eval {
    int op;
    bool is_binary;
    const TypedConstant &lhs;
    const TypedConstant &rhs;
    DataType ret_type;
    Stmt *replaced_stmt;
  };

  std::vector<deferred_jit_eval> batched_eval;

  struct DataTypeHasher {
    size_t operator()(const DataType &dt) const {
      return dt.hash();
    }
  };

  void evaluate_batched(size_t eval_kernel_id) {
    if (!batched_eval.size())
      return;

    auto kernel_name = fmt::format("jit_evaluator_{}", eval_kernel_id);

    size_t arr_size = batched_eval.size() * 2;
    std::vector<int8> ext_arr_i8(arr_size);
    std::vector<int8> ext_arr_u8(arr_size);
    std::vector<int16> ext_arr_i16(arr_size);
    std::vector<int16> ext_arr_u16(arr_size);
    std::vector<int16> ext_arr_f16(arr_size);
    std::vector<int32> ext_arr_i32(arr_size);
    std::vector<int32> ext_arr_u32(arr_size);
    std::vector<int32> ext_arr_f32(arr_size);
    std::vector<int64> ext_arr_i64(arr_size);
    std::vector<int64> ext_arr_u64(arr_size);
    std::vector<int64> ext_arr_f64(arr_size);

    auto write_typed_val = [&](const TypedConstant &c, int index) {
      if (c.dt == PrimitiveType::i8) {
        ext_arr_i8[index] = c.val_i8;
      } else if (c.dt == PrimitiveType::u8) {
        ext_arr_u8[index] = c.val_i8;
      } else if (c.dt == PrimitiveType::i16) {
        ext_arr_i16[index] = c.val_i16;
      } else if (c.dt == PrimitiveType::u16) {
        ext_arr_u16[index] = c.val_i16;
      } else if (c.dt == PrimitiveType::f16) {
        ext_arr_f16[index] = c.val_i16;
      } else if (c.dt == PrimitiveType::i32) {
        ext_arr_i32[index] = c.val_i32;
      } else if (c.dt == PrimitiveType::u32) {
        ext_arr_u32[index] = c.val_i32;
      } else if (c.dt == PrimitiveType::f32) {
        ext_arr_f32[index] = c.val_i32;
      } else if (c.dt == PrimitiveType::i64) {
        ext_arr_i64[index] = c.val_i64;
      } else if (c.dt == PrimitiveType::u64) {
        ext_arr_u64[index] = c.val_i64;
      } else {
        ext_arr_f64[index] = c.val_i64;
      }
    };

    auto read_typed_val = [&](TypedConstant &c, int index) {
      if (c.dt == PrimitiveType::i8) {
        c.val_i8 = ext_arr_i8[index];
      } else if (c.dt == PrimitiveType::u8) {
        c.val_i8 = ext_arr_u8[index];
      } else if (c.dt == PrimitiveType::i16) {
        c.val_i16 = ext_arr_i16[index];
      } else if (c.dt == PrimitiveType::u16) {
        c.val_i16 = ext_arr_u16[index];
      } else if (c.dt == PrimitiveType::f16) {
        c.val_i16 = ext_arr_f16[index];
      } else if (c.dt == PrimitiveType::i32) {
        c.val_i32 = ext_arr_i32[index];
      } else if (c.dt == PrimitiveType::u32) {
        c.val_i32 = ext_arr_u32[index];
      } else if (c.dt == PrimitiveType::f32) {
        c.val_i32 = ext_arr_f32[index];
      } else if (c.dt == PrimitiveType::i64) {
        c.val_i64 = ext_arr_i64[index];
      } else if (c.dt == PrimitiveType::u64) {
        c.val_i64 = ext_arr_u64[index];
      } else {
        c.val_i64 = ext_arr_f64[index];
      }
    };

    std::unordered_set<DataType, DataTypeHasher> dtypes;
    for (auto &e : batched_eval) {
      dtypes.insert(e.ret_type);
      dtypes.insert(e.lhs.dt);
      dtypes.insert(e.rhs.dt);
    }

    auto func = [&]() {
      std::unordered_map<DataType, Stmt *, DataTypeHasher> ext_ptr_arg_stmts;
      int arg_id = 0;
      for (auto dt : dtypes) {
        auto ext_ptr_arg_stmt_unique =
            Stmt::make<ArgLoadStmt>(arg_id++, dt, /*is_ptr=*/true);
        ext_ptr_arg_stmts[dt] = ext_ptr_arg_stmt_unique.get();
        current_ast_builder().insert(std::move(ext_ptr_arg_stmt_unique));
      }

      int param_count = 0;
      for (auto &e : batched_eval) {
        auto ret_index_stmt =
            Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(param_count));
        auto lhs_index_stmt =
            Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(param_count++));
        auto rhs_index_stmt =
            Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(param_count++));

        auto ret_ptr_stmt = Stmt::make<ExternalPtrStmt>(
            LaneAttribute(ext_ptr_arg_stmts.at(e.ret_type)),
            std::vector<Stmt *>{ret_index_stmt.get()});
        auto lhs_ptr_stmt = Stmt::make<ExternalPtrStmt>(
            LaneAttribute(ext_ptr_arg_stmts.at(e.lhs.dt)),
            std::vector<Stmt *>{lhs_index_stmt.get()});
        auto rhs_ptr_stmt = Stmt::make<ExternalPtrStmt>(
            LaneAttribute(ext_ptr_arg_stmts.at(e.rhs.dt)),
            std::vector<Stmt *>{rhs_index_stmt.get()});

        auto lhstmt = Stmt::make_typed<GlobalLoadStmt>(lhs_ptr_stmt.get());
        auto rhstmt = Stmt::make_typed<GlobalLoadStmt>(rhs_ptr_stmt.get());

        pStmt oper;
        if (e.is_binary) {
          oper = Stmt::make<BinaryOpStmt>((BinaryOpType)e.op, lhstmt.get(),
                                          rhstmt.get());
        } else {
          oper = Stmt::make<UnaryOpStmt>((UnaryOpType)e.op, lhstmt.get());
          if (unary_op_is_cast((UnaryOpType)e.op)) {
            oper->cast<UnaryOpStmt>()->cast_type = e.ret_type;
          }
        }
        auto writeback =
            Stmt::make<GlobalStoreStmt>(ret_ptr_stmt.get(), oper.get());

        current_ast_builder().insert(std::move(ret_index_stmt));
        current_ast_builder().insert(std::move(ret_ptr_stmt));
        current_ast_builder().insert(std::move(lhs_index_stmt));
        current_ast_builder().insert(std::move(lhs_ptr_stmt));
        current_ast_builder().insert(std::move(lhstmt));
        if (e.is_binary) {
          current_ast_builder().insert(std::move(rhs_index_stmt));
          current_ast_builder().insert(std::move(rhs_ptr_stmt));
          current_ast_builder().insert(std::move(rhstmt));
        }
        current_ast_builder().insert(std::move(oper));
        current_ast_builder().insert(std::move(writeback));
      }
    };

    auto program_compile_config_org = program->config;
    program->config.advanced_optimization = false;
    program->config.cfg_optimization = false;
    program->config.simplify_before_lower_access = false;
    program->config.simplify_after_lower_access = false;
    program->config.constant_folding = false;
    program->config.external_optimization_level = 0;

    auto ker = std::make_unique<Kernel>(*program, func, kernel_name);
    for (auto dt : dtypes) {
      ker->insert_arg(dt, true);
    }
    ker->is_evaluator = true;

    //std::string output;
    //irpass::print(ker->ir.get(), &output);
    //std::cout << output << std::flush;

    int i = 0;
    for (auto &e : batched_eval) {
      write_typed_val(e.lhs, i);
      if (e.is_binary)
        write_typed_val(e.rhs, i + 1);
      i += 2;
    }

    auto launch_ctx = ker->make_launch_context();

    int arg_id = 0;
    for (auto dt : dtypes) {
      if (dt == PrimitiveType::i8) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_i8.data()),
                                          uint64(arr_size));
      } else if (dt == PrimitiveType::u8) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_u8.data()),
                                          uint64(arr_size));
      } else if (dt == PrimitiveType::i16) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_i16.data()),
                                          uint64(arr_size * 2));
      } else if (dt == PrimitiveType::u16) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_u16.data()),
                                          uint64(arr_size * 2));
      } else if (dt == PrimitiveType::f16) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_f16.data()),
                                          uint64(arr_size * 2));
      } else if (dt == PrimitiveType::i32) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_i32.data()),
                                          uint64(arr_size * 4));
      } else if (dt == PrimitiveType::u32) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_u32.data()),
                                          uint64(arr_size * 4));
      } else if (dt == PrimitiveType::f32) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_f32.data()),
                                          uint64(arr_size * 4));
      } else if (dt == PrimitiveType::i64) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_i64.data()),
                                          uint64(arr_size * 8));
      } else if (dt == PrimitiveType::u64) {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_u64.data()),
                                          uint64(arr_size * 8));
      } else {
        launch_ctx.set_arg_external_array(arg_id, uint64(ext_arr_f64.data()),
                                          uint64(arr_size * 8));
      }
      arg_id++;
    }

    (*ker)(launch_ctx);

    i = 0;
    for (auto &e : batched_eval) {
      TypedConstant new_constant(e.ret_type);
      read_typed_val(new_constant, i);
      TI_TRACE("{} result = {}", e.replaced_stmt->id,
               new_constant.val_cast_to_float64());
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      e.replaced_stmt->replace_with(evaluated.get());
      modifier.insert_before(e.replaced_stmt, std::move(evaluated));
      modifier.erase(e.replaced_stmt);
      i += 2;
    }

    program->config = program_compile_config_org;

    batched_eval.clear();
  }

  static bool is_good_type(DataType dt) {
    // ConstStmt of `bad` types like `i8` is not supported by LLVM.
    // Discussion:
    // https://github.com/taichi-dev/taichi/pull/839#issuecomment-625902727
    if (dt->is_primitive(PrimitiveTypeID::i32) ||
        dt->is_primitive(PrimitiveTypeID::i64) ||
        dt->is_primitive(PrimitiveTypeID::u32) ||
        dt->is_primitive(PrimitiveTypeID::u64) ||
        dt->is_primitive(PrimitiveTypeID::f32) ||
        dt->is_primitive(PrimitiveTypeID::f64))
      return true;
    else
      return false;
  }

  bool jit_evaluate_binary_op(DataType ret_type,
                              BinaryOpStmt *stmt,
                              const TypedConstant &lhs,
                              const TypedConstant &rhs) {
    if (!is_good_type(ret_type))
      return false;

    TI_TRACE("JIT Binary {} {} {}", stmt->id, lhs.val_cast_to_float64(),
             rhs.val_cast_to_float64());
    batched_eval.push_back(
        {(int)stmt->op_type, true, lhs, rhs, ret_type, stmt});

    return true;
  }

  bool jit_evaluate_unary_op(DataType ret_type,
                             UnaryOpStmt *stmt,
                             const TypedConstant &operand) {
    if (!is_good_type(ret_type))
      return false;

    TI_TRACE("JIT Unary {} {}", stmt->id, operand.val_cast_to_float64());
    batched_eval.push_back(
        {(int)stmt->op_type, false, operand, operand, ret_type, stmt});

    return true;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs || !rhs)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type;
    jit_evaluate_binary_op(dst_type, stmt, lhs->val[0], rhs->val[0]);
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->is_cast() && stmt->cast_type == stmt->operand->ret_type) {
      stmt->replace_with(stmt->operand);
      modifier.erase(stmt);
      return;
    }
    auto operand = stmt->operand->cast<ConstStmt>();
    if (!operand)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type;
    jit_evaluate_unary_op(dst_type, stmt, operand->val[0]);
  }

  void visit(BitExtractStmt *stmt) override {
    auto input = stmt->input->cast<ConstStmt>();
    if (!input)
      return;
    if (stmt->width() != 1)
      return;
    std::unique_ptr<Stmt> result_stmt;
    if (is_signed(input->val[0].dt)) {
      auto result = (input->val[0].val_int() >> stmt->bit_begin) &
                    ((1LL << (stmt->bit_end - stmt->bit_begin)) - 1);
      result_stmt = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(
          TypedConstant(input->val[0].dt, result)));
    } else {
      auto result = (input->val[0].val_uint() >> stmt->bit_begin) &
                    ((1LL << (stmt->bit_end - stmt->bit_begin)) - 1);
      result_stmt = Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(
          TypedConstant(input->val[0].dt, result)));
    }
    stmt->replace_with(result_stmt.get());
    modifier.insert_before(stmt, std::move(result_stmt));
    modifier.erase(stmt);
  }

  static bool run(IRNode *node, Program *program) {
    ConstantFold folder(program);
    bool modified = false;
    while (true) {
      node->accept(&folder);
      folder.evaluate_batched(program->jit_evaluator_id.fetch_add(1));
      if (folder.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }
    return modified;
  }
};

const PassID ConstantFoldPass::id = "ConstantFoldPass";

namespace irpass {

bool constant_fold(IRNode *root,
                   const CompileConfig &config,
                   const ConstantFoldPass::Args &args) {
  TI_AUTO_PROF;
  // @archibate found that `debug=True` will cause JIT kernels
  // to evaluate incorrectly (always return 0), so we simply
  // disable constant_fold when config.debug is turned on.
  // Discussion:
  // https://github.com/taichi-dev/taichi/pull/839#issuecomment-626107010
  if (config.debug) {
    TI_TRACE("config.debug enabled, ignoring constant fold");
    return false;
  }
  if (!config.advanced_optimization)
    return false;
  return ConstantFold::run(root, args.program);
}

}  // namespace irpass

TLANG_NAMESPACE_END
