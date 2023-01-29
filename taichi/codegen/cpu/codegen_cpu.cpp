#include "taichi/codegen/cpu/codegen_cpu.h"

#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"
namespace taichi::lang {

namespace {

class TaskCodeGenCPU : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenCPU(const CompileConfig *config, Kernel *kernel, IRNode *ir)
      : TaskCodeGenLLVM(config, kernel, ir, nullptr) {
    TI_AUTO_PROF
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    int step = 1;

    // In parallel for-loops reversing the order doesn't make sense.
    // However, we may need to support serial offloaded range for's in the
    // future, so it still makes sense to reverse the order here.
    if (stmt->reversed) {
      step = -1;
    }

    auto *tls_prologue = create_xlogue(stmt->tls_prologue);

    // The loop body
    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::Type::getInt8PtrTy(*llvm_context),
           tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    llvm::Value *epilogue = create_xlogue(stmt->tls_epilogue);

    auto [begin, end] = get_range_for_bounds(stmt);

    // adaptive block_dim
    if (compile_config->cpu_block_dim_adaptive) {
      int num_items = (stmt->end_value - stmt->begin_value) / std::abs(step);
      int num_threads = stmt->num_cpu_threads;
      int items_per_thread = std::max(1, num_items / (num_threads * 32));
      // keep each task has at least 512 items to amortize scheduler overhead
      // also saturate the value to 1024 for better load balancing
      stmt->block_dim = std::min(1024, std::max(512, items_per_thread));
    }

    call("cpu_parallel_range_for", get_arg(0),
         tlctx->get_constant(stmt->num_cpu_threads), begin, end,
         tlctx->get_constant(step), tlctx->get_constant(stmt->block_dim),
         tls_prologue, body, epilogue, tlctx->get_constant(stmt->tls_size));
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    auto *tls_prologue = create_mesh_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::Type::getInt8PtrTy(*llvm_context),
           tlctx->get_data_type<int>()});

      for (int i = 0; i < stmt->mesh_prologue->size(); i++) {
        auto &s = stmt->mesh_prologue->statements[i];
        s->accept(this);
      }

      if (stmt->bls_prologue) {
        stmt->bls_prologue->accept(this);
      }

      auto loop_test_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_test", func);
      auto loop_body_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_body", func);
      auto func_exit =
          llvm::BasicBlock::Create(*llvm_context, "func_exit", func);
      auto loop_index =
          create_entry_block_alloca(llvm::Type::getInt32Ty(*llvm_context));
      builder->CreateStore(tlctx->get_constant(0), loop_index);
      builder->CreateBr(loop_test_bb);

      {
        builder->SetInsertPoint(loop_test_bb);
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
        auto cond = builder->CreateICmp(
            llvm::CmpInst::Predicate::ICMP_SLT, loop_index_load,
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
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
        builder->CreateStore(
            builder->CreateAdd(loop_index_load, tlctx->get_constant(1)),
            loop_index);
        builder->CreateBr(loop_test_bb);
        builder->SetInsertPoint(func_exit);
      }

      if (stmt->bls_epilogue) {
        stmt->bls_epilogue->accept(this);
      }

      body = guard.body;
    }

    llvm::Value *epilogue = create_mesh_xlogue(stmt->tls_epilogue);

    call("cpu_parallel_mesh_for", get_arg(0),
         tlctx->get_constant(stmt->num_cpu_threads),
         tlctx->get_constant(stmt->mesh->num_patches),
         tlctx->get_constant(stmt->block_dim), tls_prologue, body, epilogue,
         tlctx->get_constant(stmt->tls_size));
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                     stmt->bls_size);
    bls_buffer = new llvm::GlobalVariable(
        *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "bls_buffer", nullptr, llvm::GlobalVariable::LocalExecTLSModel, 0);
    /* module->getOrInsertGlobal("bls_buffer", type);
    bls_buffer = module->getNamedGlobal("bls_buffer");
    bls_buffer->setAlignment(llvm::MaybeAlign(8));*/ // TODO(changyu): Fix JIT session error: Symbols not found: [ __emutls_get_address ] in python 3.10

    // initialize the variable with an undef value to ensure it is added to the
    // symbol table
    bls_buffer->setInitializer(llvm::UndefValue::get(type));
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (compile_config->kernel_profiler && arch_is_cpu(compile_config->arch)) {
      call("LLVMRuntime_profiler_start", get_runtime(),
           builder->CreateGlobalStringPtr(offloaded_task_name));
    }
    if (stmt->task_type == Type::serial) {
      stmt->body->accept(this);
    } else if (stmt->task_type == Type::range_for) {
      create_offload_range_for(stmt);
    } else if (stmt->task_type == Type::mesh_for) {
      create_offload_mesh_for(stmt);
    } else if (stmt->task_type == Type::struct_for) {
      stmt->block_dim = std::min(stmt->snode->parent->max_num_elements(),
                                 (int64)stmt->block_dim);
      create_offload_struct_for(stmt, "cpu");
    } else if (stmt->task_type == Type::listgen) {
      emit_list_gen(stmt);
    } else if (stmt->task_type == Type::gc) {
      emit_gc(stmt);
    } else if (stmt->task_type == Type::gc_rc) {
      emit_gc_rc();
    } else {
      TI_NOT_IMPLEMENTED
    }
    if (compile_config->kernel_profiler && arch_is_cpu(compile_config->arch)) {
      llvm::IRBuilderBase::InsertPointGuard guard(*builder);
      builder->SetInsertPoint(final_block);
      call("LLVMRuntime_profiler_stop", get_runtime());
    }
    finalize_offloaded_task_function();
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
    current_offload = nullptr;
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      TaskCodeGenLLVM::visit_call_bitcode(stmt);
    } else if (stmt->type == ExternalFuncCallStmt::SHARED_OBJECT) {
      TaskCodeGenLLVM::visit_call_shared_object(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }
};

}  // namespace

#ifdef TI_WITH_LLVM
FunctionType CPUModuleToFunctionConverter::convert(
    const std::string &kernel_name,
    const std::vector<LlvmLaunchArgInfo> &args,
    LLVMCompiledKernel data) const {
  TI_AUTO_PROF;
  auto jit_module = tlctx_->create_jit_module(std::move(data.module));
  using TaskFunc = int32 (*)(void *);
  std::vector<TaskFunc> task_funcs;
  task_funcs.reserve(data.tasks.size());
  for (auto &task : data.tasks) {
    auto *func_ptr = jit_module->lookup_function(task.name);
    TI_ASSERT_INFO(func_ptr, "Offloaded datum function {} not found",
                   task.name);
    task_funcs.push_back((TaskFunc)(func_ptr));
  }
  // Do NOT capture `this`...
  return [executor = this->executor_, args, kernel_name,
          task_funcs](RuntimeContext &context) {
    TI_TRACE("Launching kernel {}", kernel_name);
    // For taichi ndarrays, context.args saves pointer to its
    // |DeviceAllocation|, CPU backend actually want to use the raw ptr here.
    for (int i = 0; i < (int)args.size(); i++) {
      if (args[i].is_array &&
          context.device_allocation_type[i] !=
              RuntimeContext::DevAllocType::kNone &&
          context.array_runtime_sizes[i] > 0) {
        DeviceAllocation *ptr =
            static_cast<DeviceAllocation *>(context.get_arg<void *>(i));
        uint64 host_ptr = (uint64)executor->get_ndarray_alloc_info_ptr(*ptr);
        context.set_arg(i, host_ptr);
        context.set_array_device_allocation_type(
            i, RuntimeContext::DevAllocType::kNone);

        if (context.has_grad[i]) {
          DeviceAllocation *ptr_grad =
              static_cast<DeviceAllocation *>(context.get_grad_arg<void *>(i));
          uint64 host_ptr_grad =
              (uint64)executor->get_ndarray_alloc_info_ptr(*ptr_grad);
          context.set_grad_arg(i, host_ptr_grad);
        }
      }
    }
    for (auto task : task_funcs) {
      task(&context);
    }
  };
}

LLVMCompiledTask KernelCodeGenCPU::compile_task(
    const CompileConfig *config,
    std::unique_ptr<llvm::Module> &&module,
    OffloadedStmt *stmt) {
  TaskCodeGenCPU gen(config, kernel, stmt);
  return gen.run_compilation();
}
#endif  // TI_WITH_LLVM

FunctionType KernelCodeGenCPU::compile_to_function() {
  TI_AUTO_PROF;
  auto *llvm_prog = get_llvm_program(prog);
  const auto &config = *get_compile_config();
  auto *tlctx = llvm_prog->get_llvm_context(config.arch);

  CPUModuleToFunctionConverter converter(
      tlctx, get_llvm_program(prog)->get_runtime_executor());
  return converter.convert(kernel, compile_kernel_to_module());
}
}  // namespace taichi::lang
