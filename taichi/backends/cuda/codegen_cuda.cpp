#include "codegen_cuda.h"

#include <vector>
#include <set>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/lang_util.h"
#include "taichi/backends/cuda/cuda_driver.h"
#include "taichi/codegen/codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

class CodeGenLLVMCUDA : public CodeGenLLVM {
 public:
  int kernel_grid_dim;
  int kernel_block_dim;
  int num_SMs;
  int max_block_dim;
  int saturating_num_blocks;

  using IRVisitor::visit;

  CodeGenLLVMCUDA(Kernel *kernel, IRNode *ir = nullptr)
      : CodeGenLLVM(kernel, ir) {
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().device_get_attribute(
        &num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
    CUDADriver::get_instance().device_get_attribute(
        &max_block_dim, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);

    // each SM can have 16-32 resident blocks
    saturating_num_blocks = num_SMs * 32;
#endif
  }

  FunctionType compile_module_to_executable() override {
#ifdef TI_WITH_CUDA
    auto offloaded_local = offloaded_tasks;
    for (auto &task : offloaded_local) {
      llvm::Function *func = module->getFunction(task.name);
      TI_ASSERT(func);
      tlctx->mark_function_as_cuda_kernel(func);
    }

    if (prog->config.print_kernel_llvm_ir) {
      TI_INFO("IR before global optimization");
      module->print(errs(), nullptr);
    }
    auto jit = get_current_program().llvm_context_device->jit.get();
    auto cuda_module = jit->add_module(std::move(module));

    return [offloaded_local, cuda_module,
            kernel = this->kernel](Context &context) {
      // copy data to GRAM
      auto args = kernel->args;
      std::vector<void *> host_buffers(args.size());
      std::vector<void *> device_buffers(args.size());
      bool has_buffer = false;
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_nparray) {
          has_buffer = true;
          CUDADriver::get_instance().malloc(&device_buffers[i], args[i].size);
          // replace host buffer with device buffer
          host_buffers[i] = get_current_program().context.get_arg<void *>(i);
          kernel->set_arg_nparray(i, (uint64)device_buffers[i], args[i].size);
          CUDADriver::get_instance().memcpy_host_to_device(
              (void *)device_buffers[i], host_buffers[i], args[i].size);
        }
      }
      if (has_buffer) {
        CUDADriver::get_instance().stream_synchronize(nullptr);
      }

      for (auto task : offloaded_local) {
        TI_DEBUG("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
                 task.block_dim);

        cuda_module->launch(task.name, task.grid_dim, task.block_dim,
                            {&context});
      }
      // copy data back to host
      if (has_buffer) {
        CUDADriver::get_instance().stream_synchronize(nullptr);
      }
      for (int i = 0; i < (int)args.size(); i++) {
        if (args[i].is_nparray) {
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

    auto value_type = tlctx->get_data_type(stmt->stmt->ret_type.data_type);

    std::string format;

    auto value = llvm_val[stmt->stmt];

    if (stmt->stmt->ret_type.data_type == DataType::i32) {
      format = "%d";
    } else if (stmt->stmt->ret_type.data_type == DataType::i64) {
      format = "%lld";
    } else if (stmt->stmt->ret_type.data_type == DataType::f32) {
      value_type = llvm::Type::getDoubleTy(*llvm_context);
      value = builder->CreateFPExt(value, value_type);
      format = "%f";
    } else if (stmt->stmt->ret_type.data_type == DataType::f64) {
      format = "%.12f";
    } else {
      TI_NOT_IMPLEMENTED
    }

    std::vector<llvm::Type *> types{value_type};
    auto stype = llvm::StructType::get(*llvm_context, types, false);
    auto values = builder->CreateAlloca(stype);
    auto value_ptr = builder->CreateGEP(
        values, {tlctx->get_constant(0), tlctx->get_constant(0)});
    builder->CreateStore(value, value_ptr);

    auto format_str = "[debug] " + stmt->str + " = " + format + "\n";

    llvm_val[stmt] = LLVMModuleBuilder::call(
        builder.get(), "vprintf",
        builder->CreateGlobalStringPtr(format_str, "format_string"),
        builder->CreateBitCast(values,
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
    auto loop_block_dim = stmt->block_dim;
    if (loop_block_dim == 0) {
      loop_block_dim = prog->config.default_gpu_block_dim;
    }
    kernel_grid_dim = saturating_num_blocks;
    kernel_block_dim = loop_block_dim;

    llvm::Function *body;

    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("Context"), 0),
           tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(DataType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(1), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    auto [begin, end] = get_range_for_bounds(stmt);

    create_call("gpu_parallel_range_for", {get_arg(0), begin, end, body});
  }

  void emit_cuda_gc(OffloadedStmt *stmt) {
    auto snode_id = tlctx->get_constant(stmt->snode->id);
    {
      init_offloaded_task_function(stmt, "gather_list");
      call("gc_parallel_0", get_runtime(), snode_id);
      finalize_offloaded_task_function();
      current_task->grid_dim = saturating_num_blocks;
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
      current_task->grid_dim = saturating_num_blocks;
      current_task->block_dim = 64;
      current_task->end();
      current_task = nullptr;
    }
  }

  bool kernel_argument_by_val() const override {
    return true;  // on CUDA, pass the argument by value
  }

  void visit(OffloadedStmt *stmt) override {
#if defined(TI_WITH_CUDA)
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::gc) {
      // gc has 3 kernels, so we treat it specially
      emit_cuda_gc(stmt);
      return;
    } else {
      kernel_grid_dim = 1;
      kernel_block_dim = 1;
      init_offloaded_task_function(stmt);
      if (stmt->task_type == Type::serial) {
        stmt->body->accept(this);
      } else if (stmt->task_type == Type::range_for) {
        create_offload_range_for(stmt);
      } else if (stmt->task_type == Type::struct_for) {
        kernel_grid_dim = saturating_num_blocks;
        kernel_block_dim = stmt->block_dim;
        if (kernel_block_dim == 0)
          kernel_block_dim = prog->config.default_gpu_block_dim;
        kernel_block_dim =
            std::min(stmt->snode->max_num_elements(), kernel_block_dim);
        stmt->block_dim = kernel_block_dim;
        create_offload_struct_for(stmt, true);
      } else if (stmt->task_type == Type::clear_list) {
        emit_clear_list(stmt);
      } else if (stmt->task_type == Type::listgen) {
        int branching = stmt->snode->max_num_elements();
        kernel_grid_dim = saturating_num_blocks;
        kernel_block_dim = std::min(branching, 64);
        emit_list_gen(stmt);
      } else {
        TI_NOT_IMPLEMENTED
      }
      finalize_offloaded_task_function();
      current_task->grid_dim = kernel_grid_dim;
      current_task->block_dim = kernel_block_dim;
      current_task->end();
      current_task = nullptr;
    }
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
