#include "llvm/IR/IntrinsicsDirectX.h"

#include "taichi/codegen/dx12/codegen_dx12.h"
#include "taichi/codegen/dx12/dx12_llvm_passes.h"
#include "taichi/rhi/dx12/dx12_api.h"
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

class TaskCodeGenLLVMDX12 : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenLLVMDX12(int id,
                      const CompileConfig &config,
                      TaichiLLVMContext &tlctx,
                      const Kernel *kernel,
                      IRNode *ir)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir, nullptr) {
    TI_AUTO_PROF
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
          builder->CreateIntrinsic(llvm::Intrinsic::dx_thread_id_in_group,
                                   {i32_ty}, {builder->getInt32(0)});
      // FIXME: use correct block dim.
      llvm::Value *block_dim =
          builder->getInt32(64); /*builder->CreateIntrinsic(
          llvm::Intrinsic::dx, {}, {});*/
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
    if (compile_config.kernel_profiler && arch_is_cpu(compile_config.arch)) {
      call(
          builder.get(), "LLVMRuntime_profiler_start",
          {get_runtime(), builder->CreateGlobalStringPtr(offloaded_task_name)});
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
      create_offload_struct_for(stmt);
    } else if (stmt->task_type == Type::listgen) {
      emit_list_gen(stmt);
    } else if (stmt->task_type == Type::gc) {
      emit_gc(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
    if (compile_config.kernel_profiler && arch_is_cpu(compile_config.arch)) {
      llvm::IRBuilderBase::InsertPointGuard guard(*builder);
      builder->SetInsertPoint(final_block);
      call(builder.get(), "LLVMRuntime_profiler_stop", {get_runtime()});
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

 private:
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx = tlctx->get_constant(0);
    auto block_dim = tlctx->get_constant(1);
    return std::make_tuple(thread_idx, block_dim);
  }
};

}  // namespace

#ifdef TI_WITH_LLVM

static std::vector<uint8_t> generate_dxil_from_llvm(
    LLVMCompiledTask &compiled_data,
    const CompileConfig &config,
    const taichi::lang::Kernel *kernel) {
  // generate dxil from llvm ir.
  auto offloaded_local = compiled_data.tasks;
  auto module = compiled_data.module.get();
  for (auto &task : offloaded_local) {
    llvm::Function *func = module->getFunction(task.name);
    TI_ASSERT(func);
    directx12::mark_function_as_cs_entry(func);
    directx12::set_num_threads(func, config.default_gpu_block_dim, 1, 1);
    // FIXME: save task.block_dim like
    // tlctx->mark_function_as_cuda_kernel(func, task.block_dim);
  }
  auto dx_container = directx12::global_optimize_module(module, config);
  // validate and sign dx container.
  return directx12::validate_and_sign(dx_container);
}

KernelCodeGenDX12::CompileResult KernelCodeGenDX12::compile() {
  TI_AUTO_PROF;
  const auto &config = get_compile_config();

  bool verbose = config.print_ir;
  if (kernel->is_accessor && !config.print_accessor_ir) {
    verbose = false;
  }

  irpass::compile_to_offloads(ir, config, kernel, verbose,
                              /*autodiff_mode=*/kernel->autodiff_mode,
                              /*ad_use_stack=*/true,
                              /*start_from_ast=*/kernel->ir_is_ast());

  auto block = dynamic_cast<Block *>(ir);
  TI_ASSERT(block);

  auto &offloads = block->statements;

  CompileResult Result;
  for (int i = 0; i < offloads.size(); i++) {
    auto offload = irpass::analysis::clone(offloads[i].get());
    irpass::re_id(offload.get());
    auto offload_name = offload->as<OffloadedStmt>()->task_name();

    Block blk;
    blk.insert(std::move(offload));
    auto new_data = compile_task(i, config, nullptr, &blk);

    Result.task_dxil_source_codes.emplace_back(
        generate_dxil_from_llvm(new_data, config, kernel));
    aot::CompiledOffloadedTask task;
    // FIXME: build all fields for task.
    task.name = fmt::format("{}_{}_{}", kernel->get_name(), offload_name, i);
    task.type = offload_name;
    Result.tasks.emplace_back(task);
  }
  // FIXME: set correct num_snode_trees.
  Result.num_snode_trees = 1;
  return Result;
}

LLVMCompiledTask KernelCodeGenDX12::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenLLVMDX12 gen(task_codegen_id, config, get_taichi_llvm_context(),
                          kernel, block);
  return gen.run_compilation();
}
#endif  // TI_WITH_LLVM
}  // namespace taichi::lang
