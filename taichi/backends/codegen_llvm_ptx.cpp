#include <set>
#include <taichi/common/util.h>
#include <taichi/io/io.h>

#include "../ir.h"
#include "../program.h"
#include "../tlang_util.h"
#include "codegen_cuda.h"
#include "cuda_context.h"

#if defined(TLANG_WITH_CUDA)
#include "cuda_runtime.h"
#endif

#include "codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

// NVVM IR Spec:
// https://docs.nvidia.com/cuda/archive/10.0/pdf/NVVM_IR_Specification.pdf

class CodeGenLLVMGPU : public CodeGenLLVM {
 public:
  int kernel_grid_dim;
  int kernel_block_dim;
  int num_SMs;
  int max_block_dim;
  int saturating_num_blocks;

  CodeGenLLVMGPU(CodeGenBase *codegen_base, Kernel *kernel)
      : CodeGenLLVM(codegen_base, kernel) {
#if defined(TLANG_WITH_CUDA)
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&max_block_dim, cudaDevAttrMaxBlockDimX, 0);

    // each SM can have 16-32 resident blocks
    saturating_num_blocks = num_SMs * 32;
#endif
  }

  void mark_function_as_cuda_kernel(llvm::Function *func) {
    /*******************************************************************
    Example annotation from llvm PTX doc:

    define void @kernel(float addrspace(1)* %A,
                        float addrspace(1)* %B,
                        float addrspace(1)* %C);

    !nvvm.annotations = !{!0}
    !0 = !{void (float addrspace(1)*,
                 float addrspace(1)*,
                 float addrspace(1)*)* @kernel, !"kernel", i32 1}
    *******************************************************************/

    // Mark kernel function as a CUDA __global__ function
    // Add the nvvm annotation that it is considered a kernel function.

    llvm::Metadata *md_args[] = {
        llvm::ValueAsMetadata::get(func),
        MDString::get(*llvm_context, "kernel"),
        llvm::ValueAsMetadata::get(tlctx->get_constant(1))};

    MDNode *md_node = MDNode::get(*llvm_context, md_args);

    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);
  }

  FunctionType compile_module_to_executable() override {
#if defined(TLANG_WITH_CUDA)
    auto offloaded_local = offloaded_tasks;
    for (auto &task : offloaded_local) {
      llvm::Function *func = module->getFunction(task.name);
      TC_ASSERT(func);
      mark_function_as_cuda_kernel(func);
    }

    if (prog->config.print_kernel_llvm_ir) {
      TC_INFO("IR before global optimization");
      module->print(errs(), nullptr);
    }
    auto ptx = compile_module_to_ptx(module);
    if (prog->config.print_kernel_llvm_ir_optimized) {
      TC_P(ptx);
    }
    auto cuda_module = cuda_context->compile(ptx);

    for (auto &task : offloaded_local) {
      task.cuda_func =
          (void *)cuda_context->get_function(cuda_module, task.name);
    }
    auto prog = this->prog;
    return [offloaded_local, prog](Context context) {
      for (auto task : offloaded_local) {
        if (prog->config.verbose_kernel_launches)
          TC_INFO("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
                  task.block_dim);

        if (prog->config.enable_profiler) {
          prog->profiler_llvm->start(task.name);
        }
        cuda_context->launch((CUfunction)task.cuda_func, &context,
                             task.grid_dim, task.block_dim);
        if (prog->config.enable_profiler) {
          prog->profiler_llvm->stop();
        }
      }
    };
#else
    TC_NOT_IMPLEMENTED;
    return nullptr;
#endif
  }

  void visit(PrintStmt *stmt) override {
    TC_ASSERT(stmt->width() == 1);

    auto value_type = tlctx->get_data_type(stmt->stmt->ret_type.data_type);

    std::string format;

    auto value = stmt->stmt->value;

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
      TC_NOT_IMPLEMENTED
    }

    std::vector<llvm::Type *> types{value_type};
    auto stype = llvm::StructType::get(*llvm_context, types, false);
    auto values = builder->CreateAlloca(stype);
    auto value_ptr = builder->CreateGEP(
        values, {tlctx->get_constant(0), tlctx->get_constant(0)});
    builder->CreateStore(value, value_ptr);

    auto format_str = "[debug] " + stmt->str + " = " + format + "\n";

    stmt->value = ModuleBuilder::call(
        builder, "vprintf",
        builder->CreateGlobalStringPtr(format_str, "format_string"),
        builder->CreateBitCast(values,
                               llvm::Type::getInt8PtrTy(*llvm_context)));
  }

  void emit_extra_unary(UnaryOpStmt *stmt) override {
    // functions from libdevice
    auto input = stmt->operand->value;
    auto input_taichi_type = stmt->operand->ret_type.data_type;
    auto op = stmt->op_type;

#define UNARY_STD(x)                                                        \
  else if (op == UnaryOpType::x) {                                          \
    if (input_taichi_type == DataType::f32) {                               \
      stmt->value =                                                         \
          builder->CreateCall(get_runtime_function("__nv_" #x "f"), input); \
    } else if (input_taichi_type == DataType::f64) {                        \
      stmt->value =                                                         \
          builder->CreateCall(get_runtime_function("__nv_" #x), input);     \
    } else if (input_taichi_type == DataType::i32) {                        \
      stmt->value = builder->CreateCall(get_runtime_function(#x), input);   \
    } else {                                                                \
      TC_NOT_IMPLEMENTED                                                    \
    }                                                                       \
  }
    if (op == UnaryOpType::abs) {
      if (input_taichi_type == DataType::f32) {
        stmt->value =
            builder->CreateCall(get_runtime_function("__nv_fabsf"), input);
      } else if (input_taichi_type == DataType::f64) {
        stmt->value =
            builder->CreateCall(get_runtime_function("__nv_fabs"), input);
      } else if (input_taichi_type == DataType::i32) {
        stmt->value =
            builder->CreateCall(get_runtime_function("__nv_abs"), input);
      } else {
        TC_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::sqrt) {
      if (input_taichi_type == DataType::f32) {
        stmt->value =
            builder->CreateCall(get_runtime_function("__nv_sqrtf"), input);
      } else if (input_taichi_type == DataType::f64) {
        stmt->value =
            builder->CreateCall(get_runtime_function("__nv_sqrt"), input);
      } else {
        TC_NOT_IMPLEMENTED
      }
    } else if (op == UnaryOpType::logic_not) {
      if (input_taichi_type == DataType::i32) {
        stmt->value =
            builder->CreateCall(get_runtime_function("logic_not_i32"), input);
      } else {
        TC_NOT_IMPLEMENTED
      }
    }
    UNARY_STD(exp)
    UNARY_STD(log)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(sgn)
    UNARY_STD(acos)
    UNARY_STD(asin)
    else {
      TC_P(unary_op_type_name(op));
      TC_NOT_IMPLEMENTED
    }
#undef UNARY_STD
  }

  void visit(AtomicOpStmt *stmt) override {
    TC_ASSERT(stmt->width() == 1);
    // https://llvm.org/docs/NVPTXUsage.html#address-spaces
    bool is_local = stmt->dest->is<AllocaStmt>();
    if (is_local) {
      TC_ERROR("Local atomics should have been demoted.");
    } else {
      for (int l = 0; l < stmt->width(); l++) {
        TC_ASSERT(stmt->op_type == AtomicOpType::add);
        llvm::Value *old_value;
        if (is_integral(stmt->val->ret_type.data_type)) {
          old_value = builder->CreateAtomicRMW(
              llvm::AtomicRMWInst::BinOp::Add, stmt->dest->value,
              stmt->val->value, llvm::AtomicOrdering::SequentiallyConsistent);
        } else if (stmt->val->ret_type.data_type == DataType::f32) {
          auto dt = tlctx->get_data_type(DataType::f32);
          old_value =
              builder->CreateIntrinsic(Intrinsic::nvvm_atomic_load_add_f32,
                                       {llvm::PointerType::get(dt, 0)},
                                       {stmt->dest->value, stmt->val->value});
        } else if (stmt->val->ret_type.data_type == DataType::f64) {
          auto dt = tlctx->get_data_type(DataType::f64);
          old_value =
              builder->CreateIntrinsic(Intrinsic::nvvm_atomic_load_add_f64,
                                       {llvm::PointerType::get(dt, 0)},
                                       {stmt->dest->value, stmt->val->value});
        } else {
          TC_NOT_IMPLEMENTED
        }
        stmt->value = old_value;
      }
    }
  }

  void visit(RandStmt *stmt) override {
    stmt->value =
        create_call(fmt::format("cuda_rand_{}",
                                data_type_short_name(stmt->ret_type.data_type)),
                    {get_context()});
  }
  void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void create_offload_range_for(OffloadedStmt *stmt) {
    auto loop_block_dim = stmt->block_dim;
    if (loop_block_dim == 0) {
      loop_block_dim = prog->config.default_gpu_block_dim;
    }
    kernel_grid_dim = saturating_num_blocks;
    kernel_block_dim = loop_block_dim;

    llvm::Function *body;

    {
      auto guard = get_function_creation_gurad(
          {llvm::PointerType::get(get_runtime_type("Context"), 0),
           tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(DataType::i32);
      stmt->loop_vars_llvm.push_back(loop_var);
      builder->CreateStore(get_arg(1), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    auto begin_ptr = Stmt::make<GlobalTemporaryStmt>(
        stmt->begin, VectorType(1, DataType::i32));
    auto end_ptr = Stmt::make<GlobalTemporaryStmt>(
        stmt->end, VectorType(1, DataType::i32));
    begin_ptr->accept(this);
    end_ptr->accept(this);

    create_call("gpu_parallel_range_for",
                {get_arg(0), builder->CreateLoad(begin_ptr->value, "begin"),
                 builder->CreateLoad(end_ptr->value, "end"), body});
  }

  void visit(OffloadedStmt *stmt) override {
#if defined(TLANG_WITH_CUDA)
    using Type = OffloadedStmt::TaskType;
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
          std::min(stmt->snode->parent->max_num_elements(), kernel_block_dim);
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
      TC_NOT_IMPLEMENTED
    }
    finalize_offloaded_task_function();
    current_task->grid_dim = kernel_grid_dim;
    current_task->block_dim = kernel_block_dim;
    current_task->end();
    current_task = nullptr;
#else
    TC_NOT_IMPLEMENTED
#endif
  }
};

FunctionType GPUCodeGen::codegen_llvm() {
  TC_PROFILER("cuda codegen");
  return CodeGenLLVMGPU(this, kernel).gen();
}

TLANG_NAMESPACE_END
