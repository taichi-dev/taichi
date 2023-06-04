#include "taichi/codegen/amdgpu/codegen_amdgpu.h"

#include <vector>
#include <set>
#include <functional>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"
#include "taichi/rhi/amdgpu/amdgpu_driver.h"
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/codegen/codegen_utils.h"

namespace taichi {
namespace lang {

using namespace llvm;

class TaskCodeGenAMDGPU : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;
  TaskCodeGenAMDGPU(int id,
                    const CompileConfig &config,
                    TaichiLLVMContext &tlctx,
                    const Kernel *kernel,
                    IRNode *ir = nullptr)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir) {
  }

  llvm::Value *create_print(std::string tag,
                            DataType dt,
                            llvm::Value *value) override{TI_NOT_IMPLEMENTED}

  std::tuple<llvm::Value *, llvm::Type *> create_value_and_type(
      llvm::Value *value,
      DataType dt) {
    TI_NOT_IMPLEMENTED
  }

  void visit(PrintStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    auto input = llvm_val[stmt->operand];
    auto input_taichi_type = stmt->operand->ret_type;
    auto op = stmt->op_type;

#define UNARY_STD(x)                                                    \
  else if (op == UnaryOpType::x) {                                      \
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {        \
      llvm_val[stmt] = call("__ocml_" #x "_f16", input);                \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) { \
      llvm_val[stmt] = call("__ocml_" #x "_f32", input);                \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = call("__ocml_" #x "_f64", input);                \
    } else {                                                            \
      TI_NOT_IMPLEMENTED                                                \
    }                                                                   \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {
        llvm_val[stmt] = call("__ocml_fasb_f16", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__ocml_fabs_f32", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__ocml_fabs_f64", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        auto ashr = builder->CreateAShr(input, 31);
        auto xor_i32 = builder->CreateXor(ashr, input);
        llvm_val[stmt] = builder->CreateSub(xor_i32, ashr, "", false, true);
      } else {
        TI_NOT_IMPLEMENTED
      }
    }  // TODO simplify the impl of sgn
    else if (op == UnaryOpType::sgn) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        auto ashr = builder->CreateAShr(input, 31);
        auto sub = builder->CreateSub(0, input);
        auto lshr = builder->CreateLShr(sub, 31);
        llvm_val[stmt] = builder->CreateOr(ashr, lshr);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        auto func = builder->GetInsertBlock()->getParent();
        auto bb_oeq_then = BasicBlock::Create(*llvm_context, "oeq_then", func);
        auto bb_oeq_else = BasicBlock::Create(*llvm_context, "oeq_else");
        auto bb_merge = BasicBlock::Create(*llvm_context, "merge");
        auto bb_olt_then = BasicBlock::Create(*llvm_context, "olt_then", func);
        auto bb_olt_else = BasicBlock::Create(*llvm_context, "olt_else");

        auto alloc = builder->CreateAlloca(
            llvm::Type::getFloatTy(*llvm_context), (unsigned)5);
        auto newty = llvm::PointerType::get(
            llvm::Type::getFloatTy(*llvm_context), (unsigned)0);
        auto cast = builder->CreateAddrSpaceCast(alloc, newty);
        auto fcmp_oeq = builder->CreateFCmpOEQ(
            input,
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_oeq, bb_oeq_then, bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_then);
        builder->CreateStore(
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 0),
            cast);
        builder->CreateBr(bb_merge);
        bb_oeq_then = builder->GetInsertBlock();

        func->getBasicBlockList().push_back(bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_else);
        auto fcmp_olt = builder->CreateFCmpOLT(
            input,
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_olt, bb_olt_then, bb_olt_else);
        bb_oeq_else = builder->GetInsertBlock();

        builder->SetInsertPoint(bb_olt_then);
        builder->CreateStore(
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), -1),
            cast);
        builder->CreateBr(bb_merge);
        bb_olt_then = builder->GetInsertBlock();

        func->getBasicBlockList().push_back(bb_olt_else);
        builder->SetInsertPoint(bb_olt_else);
        builder->CreateStore(
            llvm::ConstantFP::get(llvm::Type::getFloatTy(*llvm_context), 1),
            cast);
        builder->CreateBr(bb_merge);
        bb_olt_else = builder->GetInsertBlock();

        func->getBasicBlockList().push_back(bb_merge);
        builder->SetInsertPoint(bb_merge);
        llvm_val[stmt] =
            builder->CreateLoad(llvm::Type::getFloatTy(*llvm_context), cast);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        auto func = builder->GetInsertBlock()->getParent();
        auto bb_oeq_then = BasicBlock::Create(*llvm_context, "oeq_then", func);
        auto bb_oeq_else = BasicBlock::Create(*llvm_context, "oeq_else");
        auto bb_merge = BasicBlock::Create(*llvm_context, "merge");
        auto bb_olt_then = BasicBlock::Create(*llvm_context, "olt_then", func);
        auto bb_olt_else = BasicBlock::Create(*llvm_context, "olt_else");

        auto alloc = builder->CreateAlloca(
            llvm::Type::getDoubleTy(*llvm_context), (unsigned)5);
        auto newty = llvm::PointerType::get(
            llvm::Type::getDoubleTy(*llvm_context), (unsigned)0);
        auto cast = builder->CreateAddrSpaceCast(alloc, newty);
        auto fcmp_oeq = builder->CreateFCmpOEQ(
            input,
            llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_oeq, bb_oeq_then, bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_then);
        builder->CreateStore(
            llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 0),
            cast);
        builder->CreateBr(bb_merge);
        bb_oeq_then = builder->GetInsertBlock();

        func->getBasicBlockList().push_back(bb_oeq_else);
        builder->SetInsertPoint(bb_oeq_else);
        auto fcmp_olt = builder->CreateFCmpOLT(
            input,
            llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 0));
        builder->CreateCondBr(fcmp_olt, bb_olt_then, bb_olt_else);
        bb_oeq_else = builder->GetInsertBlock();

        builder->SetInsertPoint(bb_olt_then);
        builder->CreateStore(
            llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), -1),
            cast);
        builder->CreateBr(bb_merge);
        bb_olt_then = builder->GetInsertBlock();

        func->getBasicBlockList().push_back(bb_olt_else);
        builder->SetInsertPoint(bb_olt_else);
        builder->CreateStore(
            llvm::ConstantFP::get(llvm::Type::getDoubleTy(*llvm_context), 1),
            cast);
        builder->CreateBr(bb_merge);
        bb_olt_else = builder->GetInsertBlock();

        func->getBasicBlockList().push_back(bb_merge);
        builder->SetInsertPoint(bb_merge);
        llvm_val[stmt] =
            builder->CreateLoad(llvm::Type::getDoubleTy(*llvm_context), cast);
      }
    }
    UNARY_STD(cos)
    UNARY_STD(acos)
    UNARY_STD(sin)
    UNARY_STD(asin)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(exp)
    UNARY_STD(log)
    UNARY_STD(sqrt)
    else {
      TI_P(unary_op_type_name(op));
      TI_NOT_IMPLEMENTED
    }
#undef UNARY_STD
  }

  llvm::Value *optimized_reduction(AtomicOpStmt *stmt) override {
    if (!stmt->is_reduction) {
      return nullptr;
    }
    TI_ASSERT(stmt->val->ret_type->is<PrimitiveType>());
    PrimitiveTypeID prim_type =
        stmt->val->ret_type->cast<PrimitiveType>()->type;

    std::unordered_map<PrimitiveTypeID,
                       std::unordered_map<AtomicOpType, std::string>>
        fast_reductions;

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::add] = "reduce_add_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::add] = "reduce_add_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::min] = "reduce_min_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::min] = "reduce_min_f32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::max] = "reduce_max_i32";
    fast_reductions[PrimitiveTypeID::f32][AtomicOpType::max] = "reduce_max_f32";

    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_and] =
        "reduce_and_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_or] =
        "reduce_or_i32";
    fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_xor] =
        "reduce_xor_i32";

    AtomicOpType op = stmt->op_type;
    if (fast_reductions.find(prim_type) == fast_reductions.end()) {
      return nullptr;
    }
    TI_ASSERT(fast_reductions.at(prim_type).find(op) !=
              fast_reductions.at(prim_type).end());
    return call(fast_reductions.at(prim_type).at(op),
                {llvm_val[stmt->dest], llvm_val[stmt->val]});
  }

  void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           get_tls_buffer_type(), tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    auto epilogue = create_xlogue(stmt->tls_epilogue);

    auto [begin, end] = get_range_for_bounds(stmt);
    call("gpu_parallel_range_for",
         {get_arg(0), begin, end, tls_prologue, body, epilogue,
          tlctx->get_constant(stmt->tls_size)});
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    TI_NOT_IMPLEMENTED
  }

  void emit_amdgpu_gc(OffloadedStmt *stmt) {
    auto snode_id = tlctx->get_constant(stmt->snode->id);
    {
      init_offloaded_task_function(stmt, "gather_list");
      call("gc_parallel_0", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "reinit_lists");
      call("gc_parallel_1", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = 1;
      current_task->block_dim = 1;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "zero_fill");
      call("gc_parallel_2", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
  }

  bool kernel_argument_by_val() const override {
    // on AMDGPU, pass the argument by value is not allowed
    return false;
  }

  void visit(GlobalLoadStmt *stmt) override {
    auto ptr = llvm_val[stmt->src];
    auto ptr_type = stmt->src->ret_type->as<PointerType>();
    if (ptr_type->is_bit_pointer()) {
      auto val_type = ptr_type->get_pointee_type();
      auto get_ch = stmt->src->as<GetChStmt>();
      auto physical_type =
          tlctx->get_data_type(get_ch->input_snode->physical_type);
      auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
      auto physical_value = builder->CreateLoad(physical_type, byte_ptr);
      if (auto qit = val_type->cast<QuantIntType>()) {
        llvm_val[stmt] = extract_quant_int(physical_value, bit_offset, qit);
      } else if (auto qfxt = val_type->cast<QuantFixedType>()) {
        qit = qfxt->get_digits_type()->as<QuantIntType>();
        auto digits = extract_quant_int(physical_value, bit_offset, qit);
        llvm_val[stmt] = reconstruct_quant_fixed(digits, qfxt);
      } else {
        TI_ASSERT(val_type->is<QuantFloatType>());
        TI_ASSERT(get_ch->input_snode->dt->is<BitStructType>());
        llvm_val[stmt] = extract_quant_float(
            physical_value, get_ch->input_snode->dt->as<BitStructType>(),
            get_ch->output_snode->id_in_bit_struct);
      }
    } else {
      // Byte pointer case.
      llvm_val[stmt] =
          builder->CreateLoad(tlctx->get_data_type(stmt->ret_type), ptr);
    }
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    TI_NOT_IMPLEMENTED
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
#if defined(TI_WITH_AMDGPU)
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::gc) {
      emit_amdgpu_gc(stmt);
    } else {
      init_offloaded_task_function(stmt);
      if (stmt->task_type == Type::serial) {
        stmt->body->accept(this);
      } else if (stmt->task_type == Type::range_for) {
        create_offload_range_for(stmt);
      } else if (stmt->task_type == Type::struct_for) {
        create_offload_struct_for(stmt);
      } else if (stmt->task_type == Type::mesh_for) {
        create_offload_mesh_for(stmt);
      } else if (stmt->task_type == Type::listgen) {
        emit_list_gen(stmt);
      } else {
        TI_NOT_IMPLEMENTED
      }
      finalize_offloaded_task_function();
      // TODO
      // use amdgpu-jargons to replace nvidias'
      current_task->grid_dim = stmt->grid_dim;
      if (stmt->task_type == Type::range_for) {
        if (stmt->const_begin && stmt->const_end) {
          int num_threads = stmt->end_value - stmt->begin_value;
          int grid_dim = ((num_threads % stmt->block_dim) == 0)
                             ? (num_threads / stmt->block_dim)
                             : (num_threads / stmt->block_dim) + 1;
          grid_dim = std::max(grid_dim, 1);
          current_task->grid_dim = std::min(stmt->grid_dim, grid_dim);
        }
      }
      if (stmt->task_type == Type::listgen) {
        // Note: 32 is a temporary number
        // TODO: find a func to obtain this attr
        int query_max_block_per_sm = 32;
        // AMDGPUDriver::get_instance().device_get_attribute(
        //     &query_max_block_per_sm,
        //     HIP_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
        int num_SMs;
        AMDGPUDriver::get_instance().device_get_attribute(
            &num_SMs, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
        current_task->grid_dim = num_SMs * query_max_block_per_sm;
      }
      current_task->block_dim = stmt->block_dim;
      TI_ASSERT(current_task->grid_dim != 0);
      TI_ASSERT(current_task->block_dim != 0);
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    current_offload = nullptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      TaskCodeGenLLVM::visit_call_bitcode(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(BinaryOpStmt *stmt) override {
    auto op = stmt->op_type;
    auto ret_taichi_type = stmt->ret_type;
    if (op != BinaryOpType::atan2 && op != BinaryOpType::pow) {
      return TaskCodeGenLLVM::visit(stmt);
    }
    auto lhs = llvm_val[stmt->lhs];
    auto rhs = llvm_val[stmt->rhs];

    if (op == BinaryOpType::pow) {
      if (ret_taichi_type->is_primitive(PrimitiveTypeID::f16)) {
        llvm_val[stmt] = call("__ocml_pow_f16", {lhs, rhs});
      } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__ocml_pow_f32", {lhs, rhs});
      } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__ocml_pow_f64", {lhs, rhs});
      } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        auto sitofp_lhs_ =
            builder->CreateSIToFP(lhs, llvm::Type::getDoubleTy(*llvm_context));
        auto sitofp_rhs_ =
            builder->CreateSIToFP(rhs, llvm::Type::getDoubleTy(*llvm_context));
        auto ret_ = call("__ocml_pow_f64", {sitofp_lhs_, sitofp_rhs_});
        llvm_val[stmt] =
            builder->CreateFPToSI(ret_, llvm::Type::getInt32Ty(*llvm_context));
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::atan2) {
      if (ret_taichi_type->is_primitive(PrimitiveTypeID::f16)) {
        llvm_val[stmt] = call("__ocml_atan2_f16", {lhs, rhs});
      } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__ocml_atan2_f32", {lhs, rhs});
      } else if (ret_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__ocml_atan2_f64", {lhs, rhs});
      } else {
        TI_NOT_IMPLEMENTED
      }
    }
  }

 private:
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx =
        builder->CreateIntrinsic(Intrinsic::amdgcn_workitem_id_x, {}, {});
    auto workgroup_dim_ =
        call("__ockl_get_local_size",
             llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvm_context), 0));
    auto block_dim = builder->CreateTrunc(
        workgroup_dim_, llvm::Type::getInt32Ty(*llvm_context));
    return std::make_tuple(thread_idx, block_dim);
  }
};

LLVMCompiledTask KernelCodeGenAMDGPU::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenAMDGPU gen(task_codegen_id, config, get_taichi_llvm_context(),
                        kernel, block);
  return gen.run_compilation();
}

}  // namespace lang
}  // namespace taichi
