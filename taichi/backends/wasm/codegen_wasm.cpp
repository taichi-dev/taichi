#include "taichi/backends/wasm/codegen_wasm.h"

#include "taichi/codegen/codegen_llvm.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/statistics.h"
#include "taichi/util/file_sequence_writer.h"

namespace taichi {
namespace lang {

class CodeGenLLVMWASM : public CodeGenLLVM {
 public:
  using IRVisitor::visit;

  CodeGenLLVMWASM(Kernel *kernel, IRNode *ir, 
                  std::unique_ptr<llvm::Module> &&M = nullptr)
      : CodeGenLLVM(kernel, ir, std::move(M)) {
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

    auto *body = llvm::BasicBlock::Create(*llvm_context, "for_loop_body", func);
    auto *loop_inc =
        llvm::BasicBlock::Create(*llvm_context, "for_loop_inc", func);
    auto *after_loop =
        llvm::BasicBlock::Create(*llvm_context, "after_for", func);
    auto *loop_test =
        llvm::BasicBlock::Create(*llvm_context, "for_loop_test", func);

    auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
    loop_vars_llvm[stmt].push_back(loop_var);

    auto [begin, end] = get_range_for_bounds(stmt);
    if (!stmt->reversed) {
      builder->CreateStore(begin, loop_var);
    } else {
      builder->CreateStore(builder->CreateSub(end, tlctx->get_constant(1)),
                           loop_var);
    }
    builder->CreateBr(loop_test);

    {
      // test block
      builder->SetInsertPoint(loop_test);
      llvm::Value *cond;
      if (!stmt->reversed) {
        cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                   builder->CreateLoad(loop_var), end);
      } else {
        cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE,
                                   builder->CreateLoad(loop_var), begin);
      }
      builder->CreateCondBr(cond, body, after_loop);
    }

    {
      {
        builder->SetInsertPoint(body);
        stmt->body->accept(this);
      }
      builder->CreateBr(loop_inc);
      builder->SetInsertPoint(loop_inc);
      if (!stmt->reversed) {
        create_increment(loop_var, tlctx->get_constant(1));
      } else {
        create_increment(loop_var, tlctx->get_constant(-1));
      }
      builder->CreateBr(loop_test);
    }

    // next cfg
    builder->SetInsertPoint(after_loop);
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(current_offload == nullptr)
    current_offload = stmt;
    using Type = OffloadedStmt::TaskType;
    if (stmt->task_type == Type::serial) {
      stmt->body->accept(this);
    } else if (stmt->task_type == Type::range_for) {
      create_offload_range_for(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
    current_offload = nullptr;
  }

  /**
   * Extracts the original function name decorated by @ti.kernel
   *
   * @param kernel_name The format is defined in
   * https://github.com/taichi-dev/taichi/blob/734da3f8f4439ce7f6a5337df7c54fb6dc34def8/python/taichi/lang/kernel_impl.py#L360-L362
   */
  std::string extract_original_kernel_name(const std::string &kernel_name) {
    if (kernel->is_evaluator)
      return kernel_name;
    int pos = kernel_name.length() - 1;
    int underline_count = 0;
    int redundant_count = 3;
    for (; pos >= 0; --pos) {
      if (kernel_name.at(pos) == '_') {
        underline_count += 1;
        if (underline_count == redundant_count)
          break;
      }
    }
    TI_ASSERT(underline_count == redundant_count)
    return kernel_name.substr(0, pos);
  }

  std::string init_taichi_kernel_function() {
    task_function_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                {llvm::PointerType::get(context_ty, 0)}, false);

    auto task_kernel_name =
        fmt::format("{}_body", extract_original_kernel_name(kernel_name));
    func = llvm::Function::Create(task_function_type,
                                  llvm::Function::ExternalLinkage,
                                  task_kernel_name, module.get());

    for (auto &arg : func->args()) {
      kernel_args.push_back(&arg);
    }
    kernel_args[0]->setName("context");

    // entry_block has all the allocas
    this->entry_block = llvm::BasicBlock::Create(*llvm_context, "entry", func);

    // The real function body
    func_body_bb = llvm::BasicBlock::Create(*llvm_context, "body", func);
    builder->SetInsertPoint(func_body_bb);
    return task_kernel_name;
  }

  void finalize_taichi_kernel_function() {
    builder->CreateRetVoid();

    // entry_block should jump to the body after all allocas are inserted
    builder->SetInsertPoint(entry_block);
    builder->CreateBr(func_body_bb);

    if (prog->config.print_kernel_llvm_ir) {
      static FileSequenceWriter writer(
          "taichi_kernel_generic_llvm_ir_{:04d}.ll",
          "unoptimized LLVM IR (generic)");
      writer.write(module.get());
    }
    TI_ASSERT(!llvm::verifyFunction(*func, &llvm::errs()));
  }

  // This is unused
  std::string create_taichi_get_root_address_function() {
    auto task_function_type =
        llvm::FunctionType::get(llvm::Type::getInt32Ty(*llvm_context),
                                {llvm::PointerType::get(context_ty, 0)}, false);
    auto task_kernel_name = fmt::format("get_root_address");
    auto func = llvm::Function::Create(task_function_type,
                                       llvm::Function::ExternalLinkage,
                                       task_kernel_name, module.get());

    std::vector<llvm::Value *> kernel_args;
    for (auto &arg : func->args()) {
      kernel_args.push_back(&arg);
    }
    kernel_args[0]->setName("context");

    auto entry_block = llvm::BasicBlock::Create(*llvm_context, "entry", func);
    auto func_body_bb = llvm::BasicBlock::Create(*llvm_context, "body", func);
    builder->SetInsertPoint(func_body_bb);

    // memory reserved for Context object shouldn't be polluted
    llvm::Value *runtime_ptr =
        create_call("Context_get_runtime", {kernel_args[0]});
    llvm::Value *runtime = builder->CreateBitCast(
        runtime_ptr,
        llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0));
    llvm::Value *root_ptr = create_call("LLVMRuntime_get_ptr_root", {runtime});
    llvm::Value *root_address = builder->CreatePtrToInt(
        root_ptr, llvm::Type::getInt32Ty(*llvm_context));
    builder->CreateRet(root_address);

    builder->SetInsertPoint(entry_block);
    builder->CreateBr(func_body_bb);

    TI_ASSERT(!llvm::verifyFunction(*func, &llvm::errs()));
    return task_kernel_name;
  }

  //  Context's address is pass by kernel_args[0] which is supposed to be 0 in
  //  default. Runtime's address will be set to kernel_args[0] after set_root()
  //  call. The objects of Context and Runtime are overlapped with each other.
  //
  //     Context          Runtime            Root Buffer
  //     +-----------+    +-------------+    +-------------+
  //     |runtime*   |    |     ...     |    |     ...     |
  //     |arg0       |    |     ...     |    +-------------+
  //     |arg1       |    |root buffer* |
  //     |    ...    |    |     ...     |
  //     +-----------+    +-------------+
  std::string create_taichi_set_root_function() {
    auto task_function_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                {llvm::PointerType::get(context_ty, 0),
                                 llvm::Type::getInt32Ty(*llvm_context)},
                                false);
    const std::string task_kernel_name = "set_root";
    auto func = llvm::Function::Create(task_function_type,
                                       llvm::Function::ExternalLinkage,
                                       task_kernel_name, module.get());

    std::vector<llvm::Value *> kernel_args;
    for (auto &arg : func->args()) {
      kernel_args.push_back(&arg);
    }
    kernel_args[0]->setName("context");
    kernel_args[1]->setName("root");

    auto entry_block = llvm::BasicBlock::Create(*llvm_context, "entry", func);
    auto func_body_bb = llvm::BasicBlock::Create(*llvm_context, "body", func);
    builder->SetInsertPoint(func_body_bb);

    // memory reserved for Context object shouldn't be polluted
    // set runtime address to zero
    llvm::Value *runtime_address_ptr =
        create_call("Context_get_ptr_runtime", {kernel_args[0]});
    llvm::Value *runtime_address_val_ptr = builder->CreatePointerCast(
        runtime_address_ptr, llvm::Type::getInt32PtrTy(*llvm_context));
    llvm::Value *runtime_address_val = builder->CreatePtrToInt(
        kernel_args[0], llvm::Type::getInt32Ty(*llvm_context));
    builder->CreateStore(runtime_address_val, runtime_address_val_ptr);

    llvm::Value *runtime_ptr =
        create_call("Context_get_runtime", {kernel_args[0]});
    llvm::Value *runtime = builder->CreateBitCast(
        runtime_ptr,
        llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0));

    llvm::Value *root_base_ptr = builder->CreatePointerCast(
        kernel_args[0], llvm::Type::getInt32PtrTy(*llvm_context));
    llvm::Value *root_base_val = builder->CreateLoad(root_base_ptr);
    llvm::Value *root_val = builder->CreateAdd(root_base_val, kernel_args[1]);
    llvm::Value *root_ptr = builder->CreateIntToPtr(
        root_val, llvm::Type::getInt8PtrTy(*llvm_context));
    llvm::Value *ret_ptr =
        create_call("LLVMRuntime_set_root", {runtime, root_ptr});
    builder->CreateRetVoid();

    builder->SetInsertPoint(entry_block);
    builder->CreateBr(func_body_bb);

    TI_ASSERT(!llvm::verifyFunction(*func, &llvm::errs()));
    return task_kernel_name;
  }

  FunctionType gen() override {
    TI_AUTO_PROF
    // emit_to_module
    stat.add("codegen_taichi_kernel_function");
    auto offloaded_task_name = init_taichi_kernel_function();
    ir->accept(this);
    finalize_taichi_kernel_function();

    auto get_root_address_name = create_taichi_get_root_address_function();
    auto set_root_name = create_taichi_set_root_function();

    // compile_module_to_executable
    // only keep the current func
    TaichiLLVMContext::eliminate_unused_functions(
        module.get(), [&](std::string func_name) {
          return offloaded_task_name == func_name ||
                 get_root_address_name == func_name ||
                 set_root_name == func_name;
        });
    tlctx->add_module(std::move(module));
    auto kernel_symbol = tlctx->lookup_function_pointer(offloaded_task_name);
    return [=](Context &context) {
      TI_TRACE("Launching Taichi Kernel Function");
      auto func = (int32(*)(void *))kernel_symbol;
      func(&context);
    };
  }
};

FunctionType CodeGenWASM::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMWASM(kernel, ir).gen();
}

std::unique_ptr<ModuleGenValue> CodeGenWASM::modulegen(
    std::unique_ptr<llvm::Module> &&module) {
  bool init_flag = module == nullptr;
  std::vector<std::string> name_list;

  auto gen = std::make_unique<CodeGenLLVMWASM>(kernel, ir, std::move(module));
  
  name_list.push_back(gen->init_taichi_kernel_function());
  gen->emit_to_module();
  gen->finalize_taichi_kernel_function();

  if(init_flag) {
    name_list.push_back(gen->create_taichi_get_root_address_function());
    name_list.push_back(gen->create_taichi_set_root_function());
  }

  gen->tlctx->jit->global_optimize_module(gen->module.get());

  return std::make_unique<ModuleGenValue>(gen->module, name_list);
}

}  // namespace lang
}  // namespace taichi
