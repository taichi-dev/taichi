#include "taichi/backends/cuda/codegen_cuda.h"

#include <vector>
#include <set>
#include <functional>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/lang_util.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/backends/cuda/cuda_context.h"
#include "taichi/codegen/codegen_llvm.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/util/action_recorder.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

class CodeGenLLVMCUDA : public CodeGenLLVM {
 public:
  using IRVisitor::visit;

  CodeGenLLVMCUDA(Kernel *kernel, IRNode *ir = nullptr)
      : CodeGenLLVM(kernel, ir) {
  }

  bool supports_offline_cache() const override {
    return true;
  }

  FunctionType gen() override {
    auto compiled_res = run_compilation();

    CUDAModuleToFunctionConverter converter{
        tlctx, this->kernel->program->get_llvm_program_impl()};

    return converter.convert(this->kernel, std::move(compiled_res.llvm_module),
                             std::move(compiled_res.offloaded_tasks));
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
          value_arr, {tlctx->get_constant(0), tlctx->get_constant(i)});
      builder->CreateStore(values[i], value_ptr);
    }
    return LLVMModuleBuilder::call(
        builder.get(), "vprintf",
        builder->CreateGlobalStringPtr(format, "format_string"),
        builder->CreateBitCast(value_arr,
                               llvm::Type::getInt8PtrTy(*llvm_context)));
  }

  void visit(PrintStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    TI_ASSERT_INFO(stmt->contents.size() < 32,
                   "CUDA `print()` doesn't support more than 32 entries");

    std::vector<llvm::Type *> types;
    std::vector<llvm::Value *> values;

    std::string formats;
    for (auto const &content : stmt->contents) {
      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);

        formats += data_type_format(arg_stmt->ret_type);

        auto value_type = tlctx->get_data_type(arg_stmt->ret_type);
        auto value = llvm_val[arg_stmt];
        if (arg_stmt->ret_type->is_primitive(PrimitiveTypeID::f32) ||
            arg_stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
          value_type = tlctx->get_data_type(PrimitiveType::f64);
          value = builder->CreateFPExt(value, value_type);
        }

        types.push_back(value_type);
        values.push_back(value);
      } else {
        auto arg_str = std::get<std::string>(content);

        auto value = builder->CreateGlobalStringPtr(arg_str, "content_string");
        auto char_type =
            llvm::Type::getInt8Ty(*tlctx->get_this_thread_context());
        auto value_type = llvm::PointerType::get(char_type, 0);

        types.push_back(value_type);
        values.push_back(value);
        formats += "%s";
      }
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
      llvm_val[stmt] = create_call("__nv_" #x "f", input);              \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = create_call("__nv_" #x, input);                  \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) { \
      llvm_val[stmt] = create_call(#x, input);                          \
    } else {                                                            \
      TI_NOT_IMPLEMENTED                                                \
    }                                                                   \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = create_call("__nv_fabsf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = create_call("__nv_fabs", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = create_call("__nv_abs", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = create_call("__nv_sqrtf", input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = create_call("__nv_sqrt", input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::logic_not) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = create_call("logic_not_i32", input);
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
    return create_call(fast_reductions.at(prim_type).at(op),
                       {llvm_val[stmt->dest], llvm_val[stmt->val]});
  }

  // A huge hack for supporting f16 atomic add/max/min! Borrowed from
  // https://github.com/tensorflow/tensorflow/blob/470d58a83470f8ede3beaa584e6992bc71b7baa6/tensorflow/compiler/xla/service/gpu/ir_emitter.cc#L378-L490
  // The reason is that LLVM10 does not support generating atomicCAS for f16 on
  // NVPTX backend.
  //
  // Implements atomic binary operations using atomic compare-and-swap
  // (atomicCAS) as follows:
  //   1. Reads the value from the memory pointed to by output_address and
  //     records it as old_output.
  //   2. Uses old_output as one of the source operand to perform the binary
  //     operation and stores the result in new_output.
  //   3. Calls atomicCAS which implements compare-and-swap as an atomic
  //     operation. In particular, atomicCAS reads the value from the memory
  //     pointed to by output_address, and compares the value with old_output.
  //     If the two values equal, new_output is written to the same memory
  //     location and true is returned to indicate that the atomic operation
  //     succeeds. Otherwise, the new value read from the memory is returned. In
  //     this case, the new value is copied to old_output, and steps 2. and 3.
  //     are repeated until atomicCAS succeeds.
  //
  // int32 is used for the atomicCAS operation. So atomicCAS reads and writes 32
  // bit values from the memory, which is larger than the memory size required
  // by the original atomic binary operation. We mask off the last two bits of
  // the output_address and use the result as an address to read the 32 bit
  // values from the memory.
  //
  // This can avoid out of bound memory accesses, based on the assumption:
  // All buffers are 4 byte aligned and have a size of 4N.
  //
  // The pseudo code is shown below.
  //
  //   cas_new_output_address = alloca(32);
  //   cas_old_output_address = alloca(32);
  //   atomic_address = output_address & ((int64)(-4));
  //   new_output_address = cas_new_output_address + (output_address & 3);
  //
  //   *cas_old_output_address = *atomic_address;
  //   do {
  //     *cas_new_output_address = *cas_old_output_address;
  //     *new_output_address = operation(*new_output_address, *source_address);
  //     (*cas_old_output_address, success) =
  //       atomicCAS(atomic_address, *cas_old_output_address,
  //       *cas_new_output_address);
  //   } while (!success);
  //
  // TODO(sjwsl): Try to rewrite this after upgrading LLVM or supporting raw
  // NVPTX

  llvm::Value *atomic_op_using_cas(
      llvm::Value *output_address,
      llvm::Value *val,
      std::function<llvm::Value *(llvm::Value *, llvm::Value *)> op) override {
    llvm::PointerType *output_address_type =
        llvm::dyn_cast<llvm::PointerType>(output_address->getType());
    TI_ASSERT(output_address_type != nullptr);

    // element_type is the data type for the binary operation.
    llvm::Type *element_type = output_address_type->getPointerElementType();
    llvm::Type *element_address_type = element_type->getPointerTo();

    int atomic_size = 32;
    llvm::Type *atomic_type = builder->getIntNTy(atomic_size);
    llvm::Type *atomic_address_type = atomic_type->getPointerTo(
        output_address_type->getPointerAddressSpace());

    // cas_old_output_address and cas_new_output_address point to the scratch
    // memory where we store the old and new values for the repeated atomicCAS
    // operations.
    llvm::Value *cas_old_output_address =
        builder->CreateAlloca(atomic_type, nullptr);
    llvm::Value *cas_new_output_address =
        builder->CreateAlloca(atomic_type, nullptr);

    llvm::Value *atomic_memory_address;
    // binop_output_address points to the scratch memory that stores the
    // result of the binary operation.
    llvm::Value *binop_output_address;

    // Calculate bin_output_address output_address
    llvm::Type *address_int_type =
        module->getDataLayout().getIntPtrType(output_address_type);
    atomic_memory_address =
        builder->CreatePtrToInt(output_address, address_int_type);
    llvm::Value *mask = llvm::ConstantInt::get(address_int_type, 3);
    llvm::Value *offset = builder->CreateAnd(atomic_memory_address, mask);
    mask = llvm::ConstantInt::get(address_int_type, -4);
    atomic_memory_address = builder->CreateAnd(atomic_memory_address, mask);
    atomic_memory_address =
        builder->CreateIntToPtr(atomic_memory_address, atomic_address_type);
    binop_output_address = builder->CreateAdd(
        builder->CreatePtrToInt(cas_new_output_address, address_int_type),
        offset);
    binop_output_address =
        builder->CreateIntToPtr(binop_output_address, element_address_type);

    // Use the value from the memory that atomicCAS operates on to initialize
    // cas_old_output.
    llvm::Value *cas_old_output =
        builder->CreateLoad(atomic_memory_address, "cas_old_output");
    builder->CreateStore(cas_old_output, cas_old_output_address);

    llvm::BasicBlock *loop_body_bb =
        BasicBlock::Create(*llvm_context, "atomic_op_loop_body", func);
    llvm::BasicBlock *loop_exit_bb =
        BasicBlock::Create(*llvm_context, "loop_exit_bb", func);
    builder->CreateBr(loop_body_bb);
    builder->SetInsertPoint(loop_body_bb);

    // loop body for one atomicCAS
    {
      // Use cas_old_output to initialize cas_new_output.
      cas_old_output =
          builder->CreateLoad(cas_old_output_address, "cas_old_output");
      builder->CreateStore(cas_old_output, cas_new_output_address);

      auto binop_output = op(builder->CreateLoad(binop_output_address), val);
      builder->CreateStore(binop_output, binop_output_address);

      llvm::Value *cas_new_output =
          builder->CreateLoad(cas_new_output_address, "cas_new_output");

      // Emit code to perform the atomicCAS operation
      // (cas_old_output, success) = atomicCAS(memory_address, cas_old_output,
      //                                       cas_new_output);
      llvm::Value *ret_value = builder->CreateAtomicCmpXchg(
          atomic_memory_address, cas_old_output, cas_new_output,
          llvm::AtomicOrdering::SequentiallyConsistent,
          llvm::AtomicOrdering::SequentiallyConsistent);

      // Extract the memory value returned from atomicCAS and store it as
      // cas_old_output.
      builder->CreateStore(
          builder->CreateExtractValue(ret_value, 0, "cas_old_output"),
          cas_old_output_address);
      // Extract the success bit returned from atomicCAS and generate a
      // conditional branch on the success bit.
      builder->CreateCondBr(
          builder->CreateExtractValue(ret_value, 1, "success"), loop_exit_bb,
          loop_body_bb);
    }

    builder->SetInsertPoint(loop_exit_bb);

    return output_address;
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
    create_call("gpu_parallel_range_for",
                {get_arg(0), begin, end, tls_prologue, body, epilogue,
                 tlctx->get_constant(stmt->tls_size)});
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
      auto loop_index =
          create_entry_block_alloca(llvm::Type::getInt32Ty(*llvm_context));
      llvm::Value *thread_idx =
          builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
      llvm::Value *block_dim = builder->CreateIntrinsic(
          Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});
      builder->CreateStore(thread_idx, loop_index);
      builder->CreateBr(loop_test_bb);

      {
        builder->SetInsertPoint(loop_test_bb);
        auto cond = builder->CreateICmp(
            llvm::CmpInst::Predicate::ICMP_SLT, builder->CreateLoad(loop_index),
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
            builder->CreateAdd(builder->CreateLoad(loop_index), block_dim),
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

    create_call(
        "gpu_parallel_mesh_for",
        {get_arg(0), tlctx->get_constant(stmt->mesh->num_patches), tls_prologue,
         body, tls_epilogue, tlctx->get_constant(stmt->tls_size)});
  }

  void emit_cuda_gc(OffloadedStmt *stmt) {
    auto snode_id = tlctx->get_constant(stmt->snode->id);
    {
      init_offloaded_task_function(stmt, "gather_list");
      call("gc_parallel_0", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = prog->config.saturating_grid_dim;
      current_task->block_dim = 64;
      current_task->end();
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "reinit_lists");
      call("gc_parallel_1", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = 1;
      current_task->block_dim = 1;
      current_task->end();
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "zero_fill");
      call("gc_parallel_2", get_context(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = prog->config.saturating_grid_dim;
      current_task->block_dim = 64;
      current_task->end();
      current_task = nullptr;
    }
  }

  bool kernel_argument_by_val() const override {
    return true;  // on CUDA, pass the argument by value
  }

  llvm::Value *create_intrinsic_load(const DataType &dtype,
                                     llvm::Value *data_ptr) {
    auto llvm_dtype = llvm_type(dtype);
    auto llvm_dtype_ptr = llvm::PointerType::get(llvm_type(dtype), 0);
    llvm::Intrinsic::ID intrin;
    if (is_real(dtype)) {
      intrin = llvm::Intrinsic::nvvm_ldg_global_f;
    } else {
      intrin = llvm::Intrinsic::nvvm_ldg_global_i;
    }
    return builder->CreateIntrinsic(
        intrin, {llvm_dtype, llvm_dtype_ptr},
        {data_ptr, tlctx->get_constant(data_type_size(dtype))});
  }

  void visit(GlobalLoadStmt *stmt) override {
    if (auto get_ch = stmt->src->cast<GetChStmt>(); get_ch) {
      bool should_cache_as_read_only = false;
      if (current_offload->mem_access_opt.has_flag(
              get_ch->output_snode, SNodeAccessFlag::read_only)) {
        should_cache_as_read_only = true;
      }
      if (should_cache_as_read_only) {
        auto dtype = stmt->ret_type;
        if (auto ptr_type = stmt->src->ret_type->as<PointerType>();
            ptr_type->is_bit_pointer()) {
          // Bit pointer case.
          auto val_type = ptr_type->get_pointee_type();
          Type *int_in_mem = nullptr;
          // For CustomIntType "int_in_mem" refers to the type itself;
          // for CustomFloatType "int_in_mem" refers to the CustomIntType of the
          // digits.
          if (auto cit = val_type->cast<CustomIntType>()) {
            int_in_mem = val_type;
            dtype = cit->get_physical_type();
            auto [data_ptr, bit_offset] = load_bit_pointer(llvm_val[stmt->src]);
            data_ptr = builder->CreateBitCast(data_ptr, llvm_ptr_type(dtype));
            auto data = create_intrinsic_load(dtype, data_ptr);
            llvm_val[stmt] = extract_quant_int(data, bit_offset, int_in_mem);
          } else if (val_type->cast<CustomFloatType>()) {
            // TODO: support __ldg
            llvm_val[stmt] = load_quant_fixed_or_quant_float(stmt->src);
          } else {
            TI_NOT_IMPLEMENTED;
          }
        } else {
          // Byte pointer case.
          // Issue an CUDA "__ldg" instruction so that data are cached in
          // the CUDA read-only data cache.
          llvm_val[stmt] = create_intrinsic_load(dtype, llvm_val[stmt->src]);
        }
      } else {
        CodeGenLLVM::visit(stmt);
      }
    } else {
      CodeGenLLVM::visit(stmt);
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
    stat.add("codegen_offloaded_tasks");
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
        create_offload_struct_for(stmt, true);
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
      current_task->end();
      current_task = nullptr;
    }
    current_offload = nullptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      CodeGenLLVM::visit_call_bitcode(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;
    llvm_val[stmt] = create_call("RuntimeContext_get_extra_args",
                                 {get_context(), tlctx->get_constant(arg_id),
                                  tlctx->get_constant(axis)});
  }

  void visit(BinaryOpStmt *stmt) override {
    auto op = stmt->op_type;
    if (op != BinaryOpType::atan2 && op != BinaryOpType::pow) {
      return CodeGenLLVM::visit(stmt);
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
        llvm_val[stmt] = create_call("__nv_atan2f", {lhs, rhs});
      } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = create_call("__nv_atan2", {lhs, rhs});
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else {
      if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] = create_call("__nv_powf", {lhs, rhs});
      } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] = create_call("__nv_pow", {lhs, rhs});
      } else if (ret_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] = create_call("pow_i32", {lhs, rhs});
      } else if (ret_type->is_primitive(PrimitiveTypeID::i64)) {
        llvm_val[stmt] = create_call("pow_i64", {lhs, rhs});
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
};

#ifdef TI_WITH_LLVM
// static
std::unique_ptr<CodeGenLLVM> CodeGenCUDA::make_codegen_llvm(Kernel *kernel,
                                                            IRNode *ir) {
  return std::make_unique<CodeGenLLVMCUDA>(kernel, ir);
}
#endif  // TI_WITH_LLVM

static void set_arg_external_array(RuntimeContext *ctx,
                                   const std::string &kernel_name,
                                   int arg_id,
                                   uintptr_t ptr,
                                   uint64 size,
                                   bool is_device_allocation) {
  ActionRecorder::get_instance().record(
      "set_kernel_arg_ext_ptr",
      {ActionArg("kernel_name", kernel_name), ActionArg("arg_id", arg_id),
       ActionArg("address", fmt::format("0x{:x}", ptr)),
       ActionArg("array_size_in_bytes", (int64)size)});

  ctx->set_arg(arg_id, ptr);
  ctx->set_array_runtime_size(arg_id, size);
  ctx->set_array_is_device_allocation(arg_id, is_device_allocation);
}

FunctionType CodeGenCUDA::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMCUDA(kernel, ir).gen();
}

FunctionType CUDAModuleToFunctionConverter::convert(
    const std::string &kernel_name,
    const std::vector<LlvmLaunchArgInfo> &args,
    std::unique_ptr<llvm::Module> mod,
    std::vector<OffloadedTask> &&tasks) const {
#ifdef TI_WITH_CUDA
  for (const auto &task : tasks) {
    llvm::Function *func = mod->getFunction(task.name);
    TI_ASSERT(func);
    tlctx_->mark_function_as_cuda_kernel(func, task.block_dim);
  }

  auto jit = tlctx_->jit.get();
  auto cuda_module =
      jit->add_module(std::move(mod), program_->config->gpu_max_reg);

  return [cuda_module, kernel_name, args, offloaded_tasks = tasks,
          program = this->program_](RuntimeContext &context) {
    CUDAContext::get_instance().make_current();
    std::vector<void *> arg_buffers(args.size(), nullptr);
    std::vector<void *> device_buffers(args.size(), nullptr);

    bool transferred = false;
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_array) {
        const auto arr_sz = context.array_runtime_sizes[i];
        if (arr_sz == 0) {
          continue;
        }
        arg_buffers[i] = context.get_arg<void *>(i);
        if (!context.is_device_allocations[i]) {
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
            CUDADriver::get_instance().malloc(&device_buffers[i], arr_sz);
            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_buffers[i], arg_buffers[i], arr_sz);
          } else {
            device_buffers[i] = arg_buffers[i];
          }
          // device_buffers[i] saves a raw ptr on CUDA device.
          set_arg_external_array(&context, kernel_name, i,
                                 (uint64)device_buffers[i], arr_sz,
                                 /*is_device_allocation=*/false);

        } else if (arr_sz > 0) {
          // arg_buffers[i] is a DeviceAllocation*
          // TODO: Unwraps DeviceAllocation* can be done at CodeGenLLVM since
          // it's shared by cpu and cuda.
          DeviceAllocation *ptr =
              static_cast<DeviceAllocation *>(arg_buffers[i]);
          device_buffers[i] = program->get_ndarray_alloc_info_ptr(*ptr);
          // We compare arg_buffers[i] and device_buffers[i] later to check
          // if transfer happened.
          // TODO: this logic can be improved but I'll leave it to a followup
          // PR.
          arg_buffers[i] = device_buffers[i];

          // device_buffers[i] saves the unwrapped raw ptr from arg_buffers[i]
          set_arg_external_array(&context, kernel_name, i,
                                 (uint64)device_buffers[i], arr_sz,
                                 /*is_device_allocation=*/false);
        }
      }
    }
    if (transferred) {
      CUDADriver::get_instance().stream_synchronize(nullptr);
    }

    for (auto task : offloaded_tasks) {
      TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
               task.block_dim);
      cuda_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                          {&context});
    }
    // copy data back to host
    if (transferred) {
      CUDADriver::get_instance().stream_synchronize(nullptr);
      for (int i = 0; i < (int)args.size(); i++) {
        if (device_buffers[i] != arg_buffers[i]) {
          CUDADriver::get_instance().memcpy_device_to_host(
              arg_buffers[i], (void *)device_buffers[i],
              context.array_runtime_sizes[i]);
          CUDADriver::get_instance().mem_free((void *)device_buffers[i]);
        }
      }
    }
  };
#else
  TI_ERROR("No CUDA");
  return nullptr;
#endif  // TI_WITH_CUDA
}

FunctionType CUDAModuleToFunctionConverter::convert(
    const Kernel *kernel,
    std::unique_ptr<llvm::Module> mod,
    std::vector<OffloadedTask> &&tasks) const {
  return convert(kernel->name, infer_launch_args(kernel), std::move(mod),
                 std::move(tasks));
}

TLANG_NAMESPACE_END
