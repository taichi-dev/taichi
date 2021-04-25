#include "codegen_cuda.h"

#include <vector>
#include <set>

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

  FunctionType compile_module_to_executable() override {
#ifdef TI_WITH_CUDA
    eliminate_unused_functions();

    auto offloaded_local = offloaded_tasks;
    for (auto &task : offloaded_local) {
      llvm::Function *func = module->getFunction(task.name);
      TI_ASSERT(func);
      tlctx->mark_function_as_cuda_kernel(func, task.block_dim);
    }

    auto jit = kernel->program.llvm_context_device->jit.get();
    auto cuda_module =
        jit->add_module(std::move(module), kernel->program.config.gpu_max_reg);

    return [offloaded_local, cuda_module,
            kernel = this->kernel](Context &context) {
      // copy data to GRAM
      CUDAContext::get_instance().make_current();
      auto args = kernel->args;
      std::vector<void *> host_buffers(args.size(), nullptr);
      std::vector<void *> device_buffers(args.size(), nullptr);
      bool has_buffer = false;

      // We could also use kernel->make_launch_context() to create
      // |ctx_builder|, but that implies the usage of Program's context. For the
      // sake of decoupling, let's not do that and explicitly set the context we
      // want to modify.
      Kernel::LaunchContextBuilder ctx_builder(kernel, &context);
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_external_array) {
          has_buffer = true;
          // replace host buffer with device buffer
          host_buffers[i] = context.get_arg<void *>(i);
          if (args[i].size > 0) {
            // Note: both numpy and PyTorch support arrays/tensors with zeros
            // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
            // args[i].size = 0.
            CUDADriver::get_instance().malloc(&device_buffers[i], args[i].size);
            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_buffers[i], host_buffers[i], args[i].size);
          }
          ctx_builder.set_arg_external_array(i, (uint64)device_buffers[i],
                                             args[i].size);
        }
      }
      if (has_buffer) {
        CUDADriver::get_instance().stream_synchronize(nullptr);
      }

      for (auto task : offloaded_local) {
        TI_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
                 task.block_dim);
        cuda_module->launch(task.name, task.grid_dim, task.block_dim,
                            task.shmem_bytes, {&context});
      }
      // copy data back to host
      if (has_buffer) {
        CUDADriver::get_instance().stream_synchronize(nullptr);
      }
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_external_array && args[i].size > 0) {
          CUDADriver::get_instance().memcpy_device_to_host(
              host_buffers[i], (void *)device_buffers[i], args[i].size);
          CUDADriver::get_instance().mem_free((void *)device_buffers[i]);
        }
      }
    };
#else
    TI_ERROR("No CUDA");
    return nullptr;
#endif  // TI_WITH_CUDA
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
        if (arg_stmt->ret_type->is_primitive(PrimitiveTypeID::f32)) {
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
    auto op = stmt->op_type;

#define UNARY_STD(x)                                                         \
  else if (op == UnaryOpType::x) {                                           \
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {             \
      llvm_val[stmt] =                                                       \
          builder->CreateCall(get_runtime_function("__nv_" #x "f"), input);  \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {      \
      llvm_val[stmt] =                                                       \
          builder->CreateCall(get_runtime_function("__nv_" #x), input);      \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {      \
      llvm_val[stmt] = builder->CreateCall(get_runtime_function(#x), input); \
    } else {                                                                 \
      TI_NOT_IMPLEMENTED                                                     \
    }                                                                        \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_fabsf"), input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_fabs"), input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_abs"), input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_sqrtf"), input);
      } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_sqrt"), input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::logic_not) {
      if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("logic_not_i32"), input);
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
  }

  void visit(AtomicOpStmt *stmt) override {
    // https://llvm.org/docs/NVPTXUsage.html#address-spaces
    bool is_local = stmt->dest->is<AllocaStmt>();
    if (is_local) {
      TI_ERROR("Local atomics should have been demoted.");
    }
    TI_ASSERT(stmt->width() == 1);
    for (int l = 0; l < stmt->width(); l++) {
      llvm::Value *old_value;
      auto dst_type =
          stmt->dest->ret_type->as<PointerType>()->get_pointee_type();
      if (stmt->op_type == AtomicOpType::add) {
        if (dst_type->is<PrimitiveType>() && is_integral(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Add, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (!dst_type->is<CustomFloatType>() &&
                   is_real(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::FAdd, llvm_val[stmt->dest],
              llvm_val[stmt->val], AtomicOrdering::SequentiallyConsistent);
        } else if (auto cit = dst_type->cast<CustomIntType>()) {
          old_value = atomic_add_custom_int(stmt, cit);
        } else if (auto cft = dst_type->cast<CustomFloatType>()) {
          old_value = atomic_add_custom_float(stmt, cft);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::min) {
        if (is_integral(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Min, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (stmt->val->ret_type->is_primitive(PrimitiveTypeID::f32)) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_min_f32"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else if (stmt->val->ret_type->is_primitive(PrimitiveTypeID::f64)) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_min_f64"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::max) {
        if (is_integral(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Max, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (stmt->val->ret_type->is_primitive(PrimitiveTypeID::f32)) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_max_f32"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else if (stmt->val->ret_type->is_primitive(PrimitiveTypeID::f64)) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_max_f64"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::bit_and) {
        if (is_integral(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::And, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::bit_or) {
        if (is_integral(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Or, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::bit_xor) {
        if (is_integral(stmt->val->ret_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Xor, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
      llvm_val[stmt] = old_value;
    }
  }

  void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    auto tls_prologue = create_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("Context"), 0),
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
            llvm_val[stmt] = extract_custom_int(data, bit_offset, int_in_mem);
          } else if (auto cft = val_type->cast<CustomFloatType>()) {
            // TODO: support __ldg
            llvm_val[stmt] = load_custom_float(stmt->src);
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
      } else if (stmt->task_type == Type::listgen) {
        emit_list_gen(stmt);
      } else {
        TI_NOT_IMPLEMENTED
      }
      finalize_offloaded_task_function();
      current_task->grid_dim = stmt->grid_dim;
      current_task->block_dim = stmt->block_dim;
      TI_ASSERT(current_task->grid_dim != 0);
      TI_ASSERT(current_task->block_dim != 0);
      current_task->shmem_bytes = stmt->bls_size;
      current_task->end();
      current_task = nullptr;
    }
    current_offload = nullptr;
#else
    TI_NOT_IMPLEMENTED
#endif
  }

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override {
    const auto arg_id = stmt->arg_id;
    const auto axis = stmt->axis;
    llvm_val[stmt] =
        builder->CreateCall(get_runtime_function("Context_get_extra_args"),
                            {get_context(), tlctx->get_constant(arg_id),
                             tlctx->get_constant(axis)});
  }
};

FunctionType CodeGenCUDA::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMCUDA(kernel, ir).gen();
}

TLANG_NAMESPACE_END
