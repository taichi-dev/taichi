#include "codegen_cuda.h"

#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/ir/ir.h"
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

    auto jit = get_current_program().llvm_context_device->jit.get();
    auto cuda_module = jit->add_module(std::move(module));

    return [offloaded_local, cuda_module,
            kernel = this->kernel](Context &context) {
      // copy data to GRAM
      auto args = kernel->args;
      std::vector<void *> host_buffers(args.size(), nullptr);
      std::vector<void *> device_buffers(args.size(), nullptr);
      bool has_buffer = false;
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_nparray) {
          has_buffer = true;
          // replace host buffer with device buffer
          host_buffers[i] = get_current_program().context.get_arg<void *>(i);
          if (args[i].size > 0) {
            // Note: both numpy and PyTorch support arrays/tensors with zeros
            // in shapes, e.g., shape=(0) or shape=(100, 0, 200). This makes
            // args[i].size = 0.
            CUDADriver::get_instance().malloc(&device_buffers[i], args[i].size);
            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_buffers[i], host_buffers[i], args[i].size);
          }
          kernel->set_arg_nparray(i, (uint64)device_buffers[i], args[i].size);
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
        if (args[i].is_nparray && args[i].size > 0) {
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

        formats += data_type_format(arg_stmt->ret_type.data_type);

        auto value_type = tlctx->get_data_type(arg_stmt->ret_type.data_type);
        auto value = llvm_val[arg_stmt];
        if (arg_stmt->ret_type.data_type == DataType::f32) {
          value_type = tlctx->get_data_type(DataType::f64);
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

    auto format_str = formats;
    auto stype = llvm::StructType::get(*llvm_context, types, false);
    auto value_arr = builder->CreateAlloca(stype);
    for (int i = 0; i < values.size(); i++) {
      auto value_ptr = builder->CreateGEP(
          value_arr, {tlctx->get_constant(0), tlctx->get_constant(i)});
      builder->CreateStore(values[i], value_ptr);
    }
    llvm_val[stmt] = LLVMModuleBuilder::call(
        builder.get(), "vprintf",
        builder->CreateGlobalStringPtr(format_str, "format_string"),
        builder->CreateBitCast(value_arr,
                               llvm::Type::getInt8PtrTy(*llvm_context)));
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    // functions from libdevice
    auto input = llvm_val[stmt->operand];
    auto input_taichi_type = stmt->operand->ret_type.data_type;
    auto op = stmt->op_type;

#define UNARY_STD(x)                                                         \
  else if (op == UnaryOpType::x) {                                           \
    if (input_taichi_type == DataType::f32) {                                \
      llvm_val[stmt] =                                                       \
          builder->CreateCall(get_runtime_function("__nv_" #x "f"), input);  \
    } else if (input_taichi_type == DataType::f64) {                         \
      llvm_val[stmt] =                                                       \
          builder->CreateCall(get_runtime_function("__nv_" #x), input);      \
    } else if (input_taichi_type == DataType::i32) {                         \
      llvm_val[stmt] = builder->CreateCall(get_runtime_function(#x), input); \
    } else {                                                                 \
      TI_NOT_IMPLEMENTED                                                     \
    }                                                                        \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type == DataType::f32) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_fabsf"), input);
      } else if (input_taichi_type == DataType::f64) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_fabs"), input);
      } else if (input_taichi_type == DataType::i32) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_abs"), input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type == DataType::f32) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_sqrtf"), input);
      } else if (input_taichi_type == DataType::f64) {
        llvm_val[stmt] =
            builder->CreateCall(get_runtime_function("__nv_sqrt"), input);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::logic_not) {
      if (input_taichi_type == DataType::i32) {
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
      if (stmt->op_type == AtomicOpType::add) {
        if (is_integral(stmt->val->ret_type.data_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Add, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (stmt->val->ret_type.data_type == DataType::f32) {
#if LLVM_VERSION_MAJOR >= 10
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::FAdd, llvm_val[stmt->dest],
              llvm_val[stmt->val], AtomicOrdering::SequentiallyConsistent);
#else
          auto dt = tlctx->get_data_type(DataType::f32);
          old_value = builder->CreateIntrinsic(
              Intrinsic::nvvm_atomic_load_add_f32,
              {llvm::PointerType::get(dt, 0)},
              {llvm_val[stmt->dest], llvm_val[stmt->val]});
#endif
        } else if (stmt->val->ret_type.data_type == DataType::f64) {
#if LLVM_VERSION_MAJOR >= 10
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::FAdd, llvm_val[stmt->dest],
              llvm_val[stmt->val], AtomicOrdering::SequentiallyConsistent);
#else
          auto dt = tlctx->get_data_type(DataType::f64);
          old_value = builder->CreateIntrinsic(
              Intrinsic::nvvm_atomic_load_add_f64,
              {llvm::PointerType::get(dt, 0)},
              {llvm_val[stmt->dest], llvm_val[stmt->val]});
#endif
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::min) {
        if (is_integral(stmt->val->ret_type.data_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Min, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (stmt->val->ret_type.data_type == DataType::f32) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_min_f32"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else if (stmt->val->ret_type.data_type == DataType::f64) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_min_f64"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::max) {
        if (is_integral(stmt->val->ret_type.data_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Max, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (stmt->val->ret_type.data_type == DataType::f32) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_max_f32"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else if (stmt->val->ret_type.data_type == DataType::f64) {
          old_value =
              builder->CreateCall(get_runtime_function("atomic_max_f64"),
                                  {llvm_val[stmt->dest], llvm_val[stmt->val]});
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::bit_and) {
        if (is_integral(stmt->val->ret_type.data_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::And, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::bit_or) {
        if (is_integral(stmt->val->ret_type.data_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Or, llvm_val[stmt->dest],
              llvm_val[stmt->val],
              llvm::AtomicOrdering::SequentiallyConsistent);
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else if (stmt->op_type == AtomicOpType::bit_xor) {
        if (is_integral(stmt->val->ret_type.data_type)) {
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

  void visit(RandStmt *stmt) override {
    llvm_val[stmt] =
        create_call(fmt::format("cuda_rand_{}",
                                data_type_short_name(stmt->ret_type.data_type)),
                    {get_context()});
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

      auto loop_var = create_entry_block_alloca(DataType::i32);
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
      call("gc_parallel_0", get_runtime(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = prog->config.saturating_grid_dim;
      current_task->block_dim = 64;
      current_task->end();
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "reinit_lists");
      call("gc_parallel_1", get_runtime(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = 1;
      current_task->block_dim = 1;
      current_task->end();
      current_task = nullptr;
    }
    {
      init_offloaded_task_function(stmt, "zero_fill");
      call("gc_parallel_2", get_runtime(), snode_id);
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

  void visit(GlobalLoadStmt *stmt) override {
    if (auto get_ch = stmt->ptr->cast<GetChStmt>(); get_ch) {
      bool should_cache_as_read_only = false;
      for (auto s : current_offload->scratch_opt) {
        if (s.first == 1 && get_ch->output_snode == s.second) {
          should_cache_as_read_only = true;
        }
      }
      if (should_cache_as_read_only) {
        // Issue an CUDA "__ldg" instruction so that data are cached in
        // the CUDA read-only data cache.
        auto dtype = stmt->ret_type.data_type;
        auto llvm_dtype = llvm_type(dtype);
        auto llvm_dtype_ptr = llvm::PointerType::get(llvm_type(dtype), 0);
        llvm::Intrinsic::ID intrin;
        if (is_real(dtype)) {
          intrin = llvm::Intrinsic::nvvm_ldg_global_f;
        } else {
          intrin = llvm::Intrinsic::nvvm_ldg_global_i;
        }

        llvm_val[stmt] = builder->CreateIntrinsic(
            intrin, {llvm_dtype, llvm_dtype_ptr},
            {llvm_val[stmt->ptr], tlctx->get_constant(data_type_size(dtype))});
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
#if LLVM_VERSION_MAJOR >= 10
    bls_buffer->setAlignment(llvm::MaybeAlign(8));
#else
    bls_buffer->setAlignment(8);
#endif
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
        create_offload_struct_for(stmt, true);
      } else if (stmt->task_type == Type::clear_list) {
        emit_clear_list(stmt);
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
};

FunctionType CodeGenCUDA::codegen() {
  TI_PROFILER("cuda codegen");
  return CodeGenLLVMCUDA(kernel).gen();
}

TLANG_NAMESPACE_END
