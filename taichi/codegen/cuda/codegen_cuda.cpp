#include "taichi/codegen/cuda/codegen_cuda.h"

#include <vector>
#include <set>
#include <functional>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"
#include "taichi/rhi/cuda/cuda_driver.h"
#include "taichi/rhi/cuda/cuda_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/util/action_recorder.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/codegen/codegen_utils.h"

namespace taichi::lang {

using namespace llvm;

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

class TaskCodeGenCUDA : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  explicit TaskCodeGenCUDA(const CompileConfig &config,
                           TaichiLLVMContext &tlctx,
                           Kernel *kernel,
                           IRNode *ir = nullptr)
      : TaskCodeGenLLVM(config, tlctx, kernel, ir) {
  }

  llvm::Value *create_print(std::string tag,
                            DataType dt,
                            llvm::Value *value) override {
    std::string format = data_type_format(dt);
    if (value->getType() == llvm::Type::getFloatTy(*llvm_context)) {
      value =
          builder->CreateFPExt(value, llvm::Type::getDoubleTy(*llvm_context));
    }
    return create_print("[cuda codegen debug] " + tag + " " + format + "\n",
                        {value->getType()}, {value});
  }

  llvm::Value *create_print(const std::string &format,
                            const std::vector<llvm::Type *> &types,
                            const std::vector<llvm::Value *> &values) {
    auto stype = llvm::StructType::get(*llvm_context, types, false);
    auto value_arr = builder->CreateAlloca(stype);
    for (int i = 0; i < values.size(); i++) {
      auto value_ptr = builder->CreateGEP(
          stype, value_arr, {tlctx->get_constant(0), tlctx->get_constant(i)});
      builder->CreateStore(values[i], value_ptr);
    }
    return LLVMModuleBuilder::call(
        builder.get(), "vprintf",
        builder->CreateGlobalStringPtr(format, "format_string"),
        builder->CreateBitCast(value_arr,
                               llvm::Type::getInt8PtrTy(*llvm_context)));
  }

  std::tuple<llvm::Value *, llvm::Type *> create_value_and_type(
      llvm::Value *value,
      DataType dt) {
    auto value_type = tlctx->get_data_type(dt);
    if (dt->is_primitive(PrimitiveTypeID::f32) ||
        dt->is_primitive(PrimitiveTypeID::f16)) {
      value_type = tlctx->get_data_type(PrimitiveType::f64);
      value = builder->CreateFPExt(value, value_type);
    }
    if (dt->is_primitive(PrimitiveTypeID::i8)) {
      value_type = tlctx->get_data_type(PrimitiveType::i16);
      value = builder->CreateSExt(value, value_type);
    }
    if (dt->is_primitive(PrimitiveTypeID::u8)) {
      value_type = tlctx->get_data_type(PrimitiveType::u16);
      value = builder->CreateZExt(value, value_type);
    }
    return std::make_tuple(value, value_type);
  }

  void visit(PrintStmt *stmt) override {
    TI_ASSERT_INFO(stmt->contents.size() < 32,
                   "CUDA `print()` doesn't support more than 32 entries");

    std::vector<llvm::Type *> types;
    std::vector<llvm::Value *> values;

    std::string formats;
    size_t num_contents = 0;
    for (auto const &content : stmt->contents) {
      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);

        formats += data_type_format(arg_stmt->ret_type);

        auto value = llvm_val[arg_stmt];
        auto value_type = value->getType();
        if (arg_stmt->ret_type->is<TensorType>()) {
          auto dtype = arg_stmt->ret_type->cast<TensorType>();
          num_contents += dtype->get_num_elements();
          auto elem_type = dtype->get_element_type();
          for (int i = 0; i < dtype->get_num_elements(); ++i) {
            llvm::Value *elem_value;
            if (codegen_vector_type(compile_config)) {
              TI_ASSERT(llvm::dyn_cast<llvm::VectorType>(value_type));
              elem_value = builder->CreateExtractElement(value, i);
            } else {
              TI_ASSERT(llvm::dyn_cast<llvm::ArrayType>(value_type));
              elem_value = builder->CreateExtractValue(value, i);
            }
            auto [casted_value, elem_value_type] =
                create_value_and_type(elem_value, elem_type);
            types.push_back(elem_value_type);
            values.push_back(casted_value);
          }
        } else {
          num_contents++;
          auto [val, dtype] = create_value_and_type(value, arg_stmt->ret_type);
          types.push_back(dtype);
          values.push_back(val);
        }
      } else {
        num_contents += 1;
        auto arg_str = std::get<std::string>(content);

        auto value = builder->CreateGlobalStringPtr(arg_str, "content_string");
        auto char_type =
            llvm::Type::getInt8Ty(*tlctx->get_this_thread_context());
        auto value_type = llvm::PointerType::get(char_type, 0);

        types.push_back(value_type);
        values.push_back(value);
        formats += "%s";
      }
      TI_ASSERT_INFO(num_contents < 32,
                     "CUDA `print()` doesn't support more than 32 entries");
    }

    llvm_val[stmt] = create_print(formats, types, values);
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    // functions from libdevice
    auto input = llvm_val[stmt->operand];
    auto input_taichi_type = stmt->operand->ret_type;
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {
      // Promote to f32 since we don't have f16 support for extra unary ops in
      // libdevice.
      input =
          builder->CreateFPExt(input, llvm::Type::getFloatTy(*llvm_context));
      input_taichi_type = PrimitiveType::f32;
    }

    auto op = stmt->op_type;

#define UNARY_STD(x)                                                    \
  else if (op == UnaryOpType::x) {                                      \
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {        \
      llvm_val[stmt] = call("__nv_" #x "f", input);                     \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = call("__nv_" #x, input);                         \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) { \
      llvm_val[stmt] = call(#x, input);                                 \
    } else {                                                            \
      TI_NOT_IMPLEMENTED                                                \
    }                                                                   \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_fabsf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_fabs", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("__nv_abs", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i64)) {
        llvm_val[stmt] = call("__nv_llabs", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_sqrtf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_sqrt", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::logic_not) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("logic_not_i32", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    }
    UNARY_STD(exp)
    UNARY_STD(log)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(sgn)
    UNARY_STD(acos)
    UNARY_STD(asin)
    UNARY_STD(cos)
    UNARY_STD(sin)
    else {
      TI_P(unary_op_type_name(op));
      TI_NOT_IMPLEMENTED
    }
#undef UNARY_STD
    if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      // Convert back to f16.
      llvm_val[stmt] = builder->CreateFPTrunc(
          llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
    }
  }

  // Not all reduction statements can be optimized.
  // If the operation cannot be optimized, this function returns nullptr.
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
    return call(fast_reductions.at(prim_type).at(op), llvm_val[stmt->dest],
                llvm_val[stmt->val]);
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
    call("gpu_parallel_range_for", get_arg(0), begin, end, tls_prologue, body,
         epilogue, tlctx->get_constant(stmt->tls_size));
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_mesh_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           get_tls_buffer_type(), tlctx->get_data_type<int>()});

      for (int i = 0; i < stmt->mesh_prologue->size(); i++) {
        auto &s = stmt->mesh_prologue->statements[i];
        s->accept(this);
      }

      if (stmt->bls_prologue) {
        stmt->bls_prologue->accept(this);
        call("block_barrier");  // "__syncthreads()"
      }

      auto loop_test_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_test", func);
      auto loop_body_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_body", func);
      auto func_exit =
          llvm::BasicBlock::Create(*llvm_context, "func_exit", func);
      auto i32_ty = llvm::Type::getInt32Ty(*llvm_context);
      auto loop_index = create_entry_block_alloca(i32_ty);
      llvm::Value *thread_idx =
          builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
      llvm::Value *block_dim = builder->CreateIntrinsic(
          Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});
      builder->CreateStore(thread_idx, loop_index);
      builder->CreateBr(loop_test_bb);

      {
        builder->SetInsertPoint(loop_test_bb);
        auto cond = builder->CreateICmp(
            llvm::CmpInst::Predicate::ICMP_SLT,
            builder->CreateLoad(i32_ty, loop_index),
            llvm_val[stmt->owned_num_local.find(stmt->major_from_type)
                         ->second]);
        builder->CreateCondBr(cond, loop_body_bb, func_exit);
      }

      {
        builder->SetInsertPoint(loop_body_bb);
        loop_vars_llvm[stmt].push_back(loop_index);
        for (int i = 0; i < stmt->body->size(); i++) {
          auto &s = stmt->body->statements[i];
          s->accept(this);
        }
        builder->CreateStore(
            builder->CreateAdd(builder->CreateLoad(i32_ty, loop_index),
                               block_dim),
            loop_index);
        builder->CreateBr(loop_test_bb);
        builder->SetInsertPoint(func_exit);
      }

      if (stmt->bls_epilogue) {
        call("block_barrier");  // "__syncthreads()"
        stmt->bls_epilogue->accept(this);
      }

      body = guard.body;
    }

    auto tls_epilogue = create_mesh_xlogue(stmt->tls_epilogue);

    call("gpu_parallel_mesh_for", get_arg(0),
         tlctx->get_constant(stmt->mesh->num_patches), tls_prologue, body,
         tls_epilogue, tlctx->get_constant(stmt->tls_size));
  }

  void emit_cuda_gc(OffloadedStmt *stmt) {
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

  void emit_cuda_gc_rc(OffloadedStmt *stmt) {
    {
      init_offloaded_task_function(stmt, "gather_list");
      call("gc_rc_parallel_0", get_context());
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "reinit_lists");
      call("gc_rc_parallel_1", get_context());
      finalize_offloaded_task_function();
      current_task->grid_dim = 1;
      current_task->block_dim = 1;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "zero_fill");
      call("gc_rc_parallel_2", get_context());
      finalize_offloaded_task_function();
      current_task->grid_dim = compile_config.saturating_grid_dim;
      current_task->block_dim = 64;
      offloaded_tasks.push_back(*current_task);
      current_task = nullptr;
    }
  }

  bool kernel_argument_by_val() const override {
    return true;  // on CUDA, pass the argument by value
  }

  llvm::Value *create_intrinsic_load(llvm::Value *ptr,
                                     llvm::Type *ty) override {
    // Issue an "__ldg" instruction to cache data in the read-only data cache.
    auto intrin = ty->isFloatingPointTy() ? llvm::Intrinsic::nvvm_ldg_global_f
                                          : llvm::Intrinsic::nvvm_ldg_global_i;
    return builder->CreateIntrinsic(
        intrin, {ty, llvm::PointerType::get(ty, 0)},
        {ptr, tlctx->get_constant(ty->getScalarSizeInBits())});
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (auto get_ch = stmt->src->cast<GetChStmt>()) {
      bool should_cache_as_read_only = current_offload->mem_access_opt.has_flag(
          get_ch->output_snode, SNodeAccessFlag::read_only);
      create_global_load(stmt, should_cache_as_read_only);
    } else {
      create_global_load(stmt, false);
    }
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                     stmt->bls_size);
    bls_buffer = new GlobalVariable(
        *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "bls_buffer", nullptr, llvm::GlobalVariable::NotThreadLocal,
        3 /*addrspace=shared*/);
    bls_buffer->setAlignment(llvm::MaybeAlign(8));
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
#if defined(TI_WITH_CUDA)
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::gc) {
      // gc has 3 kernels, so we treat it specially
      emit_cuda_gc(stmt);
    } else if (stmt->task_type == Type::gc_rc) {
      emit_cuda_gc_rc(stmt);
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
        int query_max_block_per_sm;
        CUDADriver::get_instance().device_get_attribute(
            &query_max_block_per_sm,
            CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
        int num_SMs;
        CUDADriver::get_instance().device_get_attribute(
            &num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
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

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;
    llvm_val[stmt] =
        call("RuntimeContext_get_extra_args", get_context(),
             tlctx->get_constant(arg_id), tlctx->get_constant(axis));
  }

  void visit(BinaryOpStmt *stmt) override {
    auto op = stmt->op_type;
    if (op != BinaryOpType::atan2 && op != BinaryOpType::pow) {
      return TaskCodeGenLLVM::visit(stmt);
    }

    auto ret_type = stmt->ret_type;

    llvm::Value *lhs = llvm_val[stmt->lhs];
    llvm::Value *rhs = llvm_val[stmt->rhs];

    // This branch contains atan2 and pow which use runtime.cpp function for
    // **real** type. We don't have f16 support there so promoting to f32 is
    // necessary.
    if (stmt->lhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      lhs = builder->CreateFPExt(lhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (stmt->rhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      rhs = builder->CreateFPExt(rhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (ret_type->is_primitive(PrimitiveTypeID::f16)) {
      ret_type = PrimitiveType::f32;
    }

    if (op == BinaryOpType::atan2) {
      if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_atan2f", lhs, rhs);
      } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_atan2", lhs, rhs);
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else {
      // Note that ret_type here cannot be integral because pow with an
      // integral exponent has been demoted in the demote_operations pass
      if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = call("__nv_powf", lhs, rhs);
      } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_pow", lhs, rhs);
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    }

    // Convert back to f16 if applicable.
    if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      llvm_val[stmt] = builder->CreateFPTrunc(
          llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
    }
  }

 private:
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx =
        builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
    auto block_dim =
        builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});
    return std::make_tuple(thread_idx, block_dim);
  }
};

LLVMCompiledTask KernelCodeGenCUDA::compile_task(
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    OffloadedStmt *stmt) {
  TaskCodeGenCUDA gen(config, get_taichi_llvm_context(), kernel, stmt);
  return gen.run_compilation();
}

FunctionType KernelCodeGenCUDA::compile_to_function() {
  TI_AUTO_PROF
  CUDAModuleToFunctionConverter converter{
      &get_taichi_llvm_context(),
      get_llvm_program(prog)->get_runtime_executor()};
  return converter.convert(this->kernel, compile_kernel_to_module());
}

FunctionType CUDAModuleToFunctionConverter::convert(
    const std::string &kernel_name,
    const std::vector<LlvmLaunchArgInfo> &args,
    LLVMCompiledKernel data) const {
  auto &mod = data.module;
  auto &tasks = data.tasks;
#ifdef TI_WITH_CUDA
  auto jit = tlctx_->jit.get();
  auto cuda_module =
      jit->add_module(std::move(mod), executor_->get_config().gpu_max_reg);

  return [cuda_module, kernel_name, args, offloaded_tasks = tasks,
          executor = this->executor_](RuntimeContext &context) {
    CUDAContext::get_instance().make_current();
    std::vector<void *> arg_buffers(args.size(), nullptr);
    std::vector<void *> device_buffers(args.size(), nullptr);
    std::vector<DeviceAllocation> temporary_devallocs(args.size());
    char *device_result_buffer{nullptr};
    CUDADriver::get_instance().malloc_async(
        (void **)&device_result_buffer,
        std::max(context.result_buffer_size, sizeof(uint64)), nullptr);

    bool transferred = false;
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_array) {
        const auto arr_sz = context.array_runtime_sizes[i];
        if (arr_sz == 0) {
          continue;
        }
        arg_buffers[i] = context.get_arg<void *>(i);
        if (context.device_allocation_type[i] ==
            RuntimeContext::DevAllocType::kNone) {
          // Note: both numpy and PyTorch support arrays/tensors with zeros
          // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
          // `arr_sz` zero.
          unsigned int attr_val = 0;
          uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
              &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
              (void *)arg_buffers[i]);

          if (ret_code != CUDA_SUCCESS || attr_val != CU_MEMORYTYPE_DEVICE) {
            // Copy to device buffer if arg is on host
            // - ret_code != CUDA_SUCCESS:
            //   arg_buffers[i] is not on device
            // - attr_val != CU_MEMORYTYPE_DEVICE:
            //   Cuda driver is aware of arg_buffers[i] but it might be on
            //   host.
            // See CUDA driver API `cuPointerGetAttribute` for more details.
            transferred = true;

            DeviceAllocation devalloc = executor->allocate_memory_ndarray(
                arr_sz, (uint64 *)device_result_buffer);
            device_buffers[i] = executor->get_ndarray_alloc_info_ptr(devalloc);
            temporary_devallocs[i] = devalloc;

            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_buffers[i], arg_buffers[i], arr_sz);
          } else {
            device_buffers[i] = arg_buffers[i];
          }
          // device_buffers[i] saves a raw ptr on CUDA device.
          context.set_arg(i, (uint64)device_buffers[i]);

        } else if (arr_sz > 0) {
          // arg_buffers[i] is a DeviceAllocation*
          // TODO: Unwraps DeviceAllocation* can be done at TaskCodeGenLLVM
          // since it's shared by cpu and cuda.
          DeviceAllocation *ptr =
              static_cast<DeviceAllocation *>(arg_buffers[i]);
          device_buffers[i] = executor->get_ndarray_alloc_info_ptr(*ptr);
          // We compare arg_buffers[i] and device_buffers[i] later to check
          // if transfer happened.
          // TODO: this logic can be improved but I'll leave it to a followup
          // PR.
          arg_buffers[i] = device_buffers[i];

          // device_buffers[i] saves the unwrapped raw ptr from arg_buffers[i]
          context.set_arg(i, (uint64)device_buffers[i]);
        }
      }
    }
    if (transferred) {
      CUDADriver::get_instance().stream_synchronize(nullptr);
    }
    char *host_result_buffer = (char *)context.result_buffer;
    if (context.result_buffer_size > 0) {
      context.result_buffer = (uint64 *)device_result_buffer;
    }
    char *device_arg_buffer = nullptr;
    if (context.arg_buffer_size > 0) {
      CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer,
                                              context.arg_buffer_size, nullptr);
      CUDADriver::get_instance().memcpy_host_to_device_async(
          device_arg_buffer, context.arg_buffer, context.arg_buffer_size,
          nullptr);
      context.arg_buffer = device_arg_buffer;
    }
    CUDADriver::get_instance().context_set_limit(
        CU_LIMIT_STACK_SIZE, executor->get_config().cuda_stack_limit);

    for (auto task : offloaded_tasks) {
      TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
               task.block_dim);
      cuda_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                          {&context}, {});
    }
    if (context.arg_buffer_size > 0) {
      CUDADriver::get_instance().mem_free_async(device_arg_buffer, nullptr);
    }
    if (context.result_buffer_size > 0) {
      CUDADriver::get_instance().memcpy_device_to_host_async(
          host_result_buffer, device_result_buffer, context.result_buffer_size,
          nullptr);
    }
    CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
    // copy data back to host
    if (transferred) {
      CUDADriver::get_instance().stream_synchronize(nullptr);
      for (int i = 0; i < (int)args.size(); i++) {
        if (device_buffers[i] != arg_buffers[i]) {
          CUDADriver::get_instance().memcpy_device_to_host(
              arg_buffers[i], (void *)device_buffers[i],
              context.array_runtime_sizes[i]);
          executor->deallocate_memory_ndarray(temporary_devallocs[i]);
        }
      }
    }
  };
#else
  TI_ERROR("No CUDA");
  return nullptr;
#endif  // TI_WITH_CUDA
}

}  // namespace taichi::lang
