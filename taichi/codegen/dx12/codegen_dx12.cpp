#include "taichi/codegen/dx12/codegen_dx12.h"

#include "dx12_llvm_passes.h"

#include "taichi/rhi/dx12/dx12_api.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"
TLANG_NAMESPACE_BEGIN

namespace {

class TaskCodeGenLLVMDX12 : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenLLVMDX12(Kernel *kernel, IRNode *ir)
      : TaskCodeGenLLVM(kernel, ir, nullptr) {
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
    create_call("gpu_parallel_range_for",
                {get_arg(0), begin, end, tls_prologue, body, epilogue,
                 tlctx->get_constant(stmt->tls_size)});
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
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
    stat.add("codegen_offloaded_tasks");
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (prog->config.kernel_profiler && arch_is_cpu(prog->config.arch)) {
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
    if (prog->config.kernel_profiler && arch_is_cpu(prog->config.arch)) {
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
};

}  // namespace

#ifdef TI_WITH_LLVM

static std::vector<uint8_t> generate_dxil_from_llvm(LLVMCompiledData &compiled_data,
                                    taichi::lang::Kernel *kernel) {
  // generate dxil from llvm ir.
  auto offloaded_local = compiled_data.tasks;
  auto module = compiled_data.module.get();
  for (auto &task : offloaded_local) {
    llvm::Function *func = module->getFunction(task.name);
    TI_ASSERT(func);
    directx12::mark_function_as_cs_entry(func);
    directx12::set_num_threads(
        func, kernel->program->config.default_gpu_block_dim, 1, 1);
    // FIXME: save task.block_dim like
    // tlctx->mark_function_as_cuda_kernel(func, task.block_dim);
  }
  auto dx_container = directx12::global_optimize_module(module, kernel->program->config);
  // validate and sign dx container.
  return directx12::validate_and_sign(dx_container);
}

KernelCodeGenDX12::CompileResult KernelCodeGenDX12::compile() {
  TI_AUTO_PROF;
  auto *llvm_prog = get_llvm_program(prog);
  auto *tlctx = llvm_prog->get_llvm_context(kernel->arch);
  auto &config = prog->config;
  std::string kernel_key = get_hashed_offline_cache_key(&config, kernel);
  kernel->set_kernel_key_for_cache(kernel_key);

  if (!kernel->lowered()) {
    kernel->lower(/*to_executable=*/false);
  }

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  TI_ASSERT(block);

  auto &offloads = block->statements;

  CompileResult Result;
  for (int i = 0; i < offloads.size(); i++) {
    auto offload =
        irpass::analysis::clone(offloads[i].get(), offloads[i]->get_kernel());
    irpass::re_id(offload.get());
    auto *offload_stmt = offload->as<OffloadedStmt>();
    auto new_data = this->modulegen(nullptr, offload_stmt);

    Result.task_dxil_source_codes.emplace_back(
        generate_dxil_from_llvm(new_data, kernel));
    aot::CompiledOffloadedTask task;
    // FIXME: build all fields for task.
    task.name = fmt::format("{}_{}_{}", kernel->get_name(),
                            offload_stmt->task_name(),
                            i);
    task.type = offload_stmt->task_name();
    Result.tasks.emplace_back(task);
  }
  // FIXME: set correct num_snode_trees.
  Result.num_snode_trees = 1;
  return Result;
}

LLVMCompiledData KernelCodeGenDX12::modulegen(
    std::unique_ptr<llvm::Module> &&module,
                                       OffloadedStmt *stmt) {
  TaskCodeGenLLVMDX12 gen(kernel, stmt);
  return gen.run_compilation();
}
#endif  // TI_WITH_LLVM

FunctionType KernelCodeGenDX12::codegen() {
  // FIXME: implement codegen.
  return [](RuntimeContext &ctx) {
  };
}
TLANG_NAMESPACE_END
