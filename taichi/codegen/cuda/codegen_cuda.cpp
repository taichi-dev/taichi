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
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/codegen/codegen_utils.h"

namespace taichi::lang {

using namespace llvm;

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

static bool is_half2(DataType dt) {
  if (dt->is<TensorType>()) {
    auto tensor_type = dt->as<TensorType>();
    return tensor_type->get_element_type() == PrimitiveType::f16 &&
           tensor_type->get_num_elements() == 2;
  }

  return false;
}

class TaskCodeGenCUDA : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;
  size_t dynamic_shared_array_bytes{0};

  explicit TaskCodeGenCUDA(int id,
                           const CompileConfig &config,
                           TaichiLLVMContext &tlctx,
                           const Kernel *kernel,
                           IRNode *ir = nullptr)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir) {
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
    if (dt->is_primitive(PrimitiveTypeID::u1)) {
      value_type = tlctx->get_data_type(PrimitiveType::i32);
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
    for (auto i = 0; i < stmt->contents.size(); ++i) {
      auto const &content = stmt->contents[i];
      auto const &format = stmt->formats[i];

      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);

        auto &&merged_format = merge_printf_specifier(
            format, data_type_format(arg_stmt->ret_type));
        // CUDA supports all conversions, but not 'F'.
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#format-specifiers
        std::replace(merged_format.begin(), merged_format.end(), 'F', 'f');
        formats += merged_format;

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

  void visit(AllocaStmt *stmt) override {
    // Override shared memory codegen logic for large shared memory
    auto tensor_type = stmt->ret_type.ptr_removed()->cast<TensorType>();
    if (tensor_type && stmt->is_shared) {
      size_t shared_array_bytes =
          tensor_type->get_num_elements() *
          data_type_size(tensor_type->get_element_type());
      if (shared_array_bytes > cuda_dynamic_shared_array_threshold_bytes) {
        if (dynamic_shared_array_bytes > 0) {
          /* Current version only allows one dynamic shared array allocation,
           * otherwise the results could be wrong.
           * However, we should be able to collect multiple user allocations
           * and transparently apply a proper offset.
           *
           * TODO: remove the limits.
           */
          TI_ERROR(
              "Only one single large shared array instance is allowed in "
              "current version.")
        }
        // Clear tensor shape for dynamic shared memory.
        tensor_type->set_shape(std::vector<int>({0}));
        dynamic_shared_array_bytes += shared_array_bytes;
      }

      auto type = tlctx->get_data_type(tensor_type);
      auto base = new llvm::GlobalVariable(
          *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
          fmt::format("shared_array_{}", stmt->id), nullptr,
          llvm::GlobalVariable::NotThreadLocal, 3 /*addrspace=shared*/);
      base->setAlignment(llvm::MaybeAlign(8));
      auto ptr_type = llvm::PointerType::get(type, 0);
      llvm_val[stmt] = builder->CreatePointerCast(base, ptr_type);
    } else {
      TaskCodeGenLLVM::visit(stmt);
    }
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
    } else if (op == UnaryOpType::frexp) {
      auto stype = tlctx->get_data_type(stmt->ret_type.ptr_removed());
      auto res = builder->CreateAlloca(stype);
      auto frac_ptr = builder->CreateStructGEP(stype, res, 0);
      auto exp_ptr = builder->CreateStructGEP(stype, res, 1);
      // __nv_frexp onlys takes in double
      auto double_input =
          input_taichi_type->is_primitive(PrimitiveTypeID::f32)
              ? builder->CreateFPExt(
                    input,
                    llvm::Type::getDoubleTy(*tlctx->get_this_thread_context()))
              : input;
      auto frac = call("__nv_frexp", double_input, exp_ptr);
      auto output =
          input_taichi_type->is_primitive(PrimitiveTypeID::f32)
              ? builder->CreateFPTrunc(
                    frac,
                    llvm::Type::getFloatTy(*tlctx->get_this_thread_context()))
              : frac;
      builder->CreateStore(output, frac_ptr);
      llvm_val[stmt] = res;
    } else if (op == UnaryOpType::popcnt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::u64) ||
          input_taichi_type->is_primitive(PrimitiveTypeID::i64)) {
        stmt->ret_type = PrimitiveType::i32;
        llvm_val[stmt] = call("__nv_popcll", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32) ||
                 input_taichi_type->is_primitive(PrimitiveTypeID::u32)) {
        llvm_val[stmt] = call("__nv_popc", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::clz) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        stmt->ret_type = PrimitiveType::i32;
        llvm_val[stmt] = call("__nv_clz", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i64)) {
        llvm_val[stmt] = call("__nv_clzll", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::log) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        // logf has fast-math option
        llvm_val[stmt] = call(
            compile_config.fast_math ? "__nv_fast_logf" : "__nv_logf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_log", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("log", input);
      } else {
        TI_ERROR("log() for type {} is not supported",
                 input_taichi_type.to_string());
      }
    } else if (op == UnaryOpType::sin) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        // sinf has fast-math option
        llvm_val[stmt] = call(
            compile_config.fast_math ? "__nv_fast_sinf" : "__nv_sinf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_sin", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("sin", input);
      } else {
        TI_ERROR("sin() for type {} is not supported",
                 input_taichi_type.to_string());
      }
    } else if (op == UnaryOpType::cos) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        // cosf has fast-math option
        llvm_val[stmt] = call(
            compile_config.fast_math ? "__nv_fast_cosf" : "__nv_cosf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = call("__nv_cos", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = call("cos", input);
      } else {
        TI_ERROR("cos() for type {} is not supported",
                 input_taichi_type.to_string());
      }
    }
    UNARY_STD(exp)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(sgn)
    UNARY_STD(acos)
    UNARY_STD(asin)
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

  void visit(AtomicOpStmt *atomic_stmt) override {
    auto dest_type = atomic_stmt->dest->ret_type.ptr_removed();
    auto val_type = atomic_stmt->val->ret_type;

    // Half2 atomic_add is supported starting from sm_60
    //
    // TODO(zhanlue): Add capability support & validation for CUDA AOT
    //
    // For now, the following code may potentially cause trouble for CUDA AOT.
    // With half2 vectorization enabled, if one compiles the code on GPU with
    // caps >= 60, then distribute it to runtime machine with GPU caps < 60,
    // it's likely gonna crash

    std::string cuda_library_path = get_custom_cuda_library_path();
    int cap = CUDAContext::get_instance().get_compute_capability();
    if (is_half2(dest_type) && is_half2(val_type) &&
        atomic_stmt->op_type == AtomicOpType::add && cap >= 60 &&
        !cuda_library_path.empty()) {
      /*
        Half2 optimization for float16 atomic add

        [CHI IR]
            TensorType<2 x f16> old_val = atomic_add(TensorType<2 x f16>
        dest_ptr*, TensorType<2 x f16> val)

        [CodeGen]
            old_val_ptr = Alloca(TensorType<2 x f16>)

            val_ptr = Alloca(TensorType<2 x f16>)
            GEP(val_ptr, 0) = ExtractValue(val, 0)
            GEP(val_ptr, 1) = ExtractValue(val, 1)

            half2_atomic_add(dest_ptr, old_val_ptr, val_ptr)

            old_val = Load(old_val_ptr)
      */
      // Allocate old_val_ptr to store the result of atomic_add
      auto char_type = llvm::Type::getInt8Ty(*tlctx->get_this_thread_context());
      auto half_type = llvm::Type::getHalfTy(*tlctx->get_this_thread_context());
      auto ptr_type = llvm::PointerType::get(char_type, 0);

      llvm::Value *old_val = builder->CreateAlloca(half_type);
      llvm::Value *old_val_ptr = builder->CreateBitCast(old_val, ptr_type);

      // Prepare dest_ptr via pointer cast
      llvm::Value *dest_half2_ptr =
          builder->CreateBitCast(llvm_val[atomic_stmt->dest], ptr_type);

      // Prepare value_ptr from val
      llvm::ArrayType *array_type = llvm::ArrayType::get(half_type, 2);
      llvm::Value *value_ptr = builder->CreateAlloca(array_type);
      llvm::Value *value_ptr0 =
          builder->CreateGEP(array_type, value_ptr,
                             {tlctx->get_constant(0), tlctx->get_constant(0)});
      llvm::Value *value_ptr1 =
          builder->CreateGEP(array_type, value_ptr,
                             {tlctx->get_constant(0), tlctx->get_constant(1)});
      llvm::Value *value0 =
          builder->CreateExtractValue(llvm_val[atomic_stmt->val], {0});
      llvm::Value *value1 =
          builder->CreateExtractValue(llvm_val[atomic_stmt->val], {1});
      builder->CreateStore(value0, value_ptr0);
      builder->CreateStore(value1, value_ptr1);
      llvm::Value *value_half2_ptr =
          builder->CreateBitCast(value_ptr, ptr_type);
      // Defined in taichi/runtime/llvm/runtime_module/cuda_runtime.cu
      call("half2_atomic_add", dest_half2_ptr, old_val_ptr, value_half2_ptr);

      llvm_val[atomic_stmt] = builder->CreateLoad(half_type, old_val);
      return;
    }

    TaskCodeGenLLVM::visit(atomic_stmt);
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

  bool kernel_argument_by_val() const override {
    return true;  // on CUDA, pass the argument by value
  }

  llvm::Value *create_intrinsic_load(llvm::Value *ptr,
                                     llvm::Type *ty) override {
    // Issue an "__ldg" instruction to cache data in the read-only data cache.
    auto intrin = ty->isFloatingPointTy() ? llvm::Intrinsic::nvvm_ldg_global_f
                                          : llvm::Intrinsic::nvvm_ldg_global_i;
    // Special treatment for bool types. As nvvm_ldg_global_i does not support
    // 1-bit integer, so we convert them to i8.
    if (ty->getScalarSizeInBits() == 1) {
      auto *new_ty = tlctx->get_data_type<uint8>();
      auto *new_ptr =
          builder->CreatePointerCast(ptr, llvm::PointerType::get(new_ty, 0));
      auto *v = builder->CreateIntrinsic(
          intrin, {new_ty, llvm::PointerType::get(new_ty, 0)},
          {new_ptr, tlctx->get_constant(new_ty->getScalarSizeInBits())});
      return builder->CreateIsNotNull(v);
    }
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
      current_task->dynamic_shared_array_bytes = dynamic_shared_array_bytes;
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
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenCUDA gen(task_codegen_id, config, get_taichi_llvm_context(),
                      kernel, block);
  return gen.run_compilation();
}

}  // namespace taichi::lang
