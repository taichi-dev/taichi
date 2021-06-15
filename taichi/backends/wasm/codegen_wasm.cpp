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

  CodeGenLLVMWASM(Kernel *kernel, IRNode *ir) : CodeGenLLVM(kernel, ir) {
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

  std::string init_taichi_kernel_function() {
    task_function_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                {llvm::PointerType::get(context_ty, 0)}, false);

    auto task_kernel_name = fmt::format("{}_body", kernel_name);
    func = llvm::Function::Create(task_function_type,
                                  llvm::Function::ExternalLinkage,
                                  task_kernel_name, module.get());

    for (auto &arg : func->args()) {
      kernel_args.push_back(&arg);
    }
    kernel_args[0]->setName("context");

    if (kernel_argument_by_val())
      func->addParamAttr(0, llvm::Attribute::ByVal);

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
      static FileSequenceWriter writer("taichi_kernel_generic_llvm_ir_{:04d}.ll",
                                       "unoptimized LLVM IR (generic)");
      writer.write(module.get());
    }
    TI_ASSERT(!llvm::verifyFunction(*func, &llvm::errs()));
    // TI_INFO("Kernel function verified.");
  }

  FunctionType gen() override {
    TI_AUTO_PROF
    // emit_to_module
    stat.add("codegen_taichi_kernel_function");
    auto offloaded_task_name = init_taichi_kernel_function();
    ir->accept(this);
    finalize_taichi_kernel_function();

    // compile_module_to_executable
    // only keep the current func
    TaichiLLVMContext::eliminate_unused_functions(
      module.get(), [&](std::string func_name) {
        return offloaded_task_name == func_name;
      });
    tlctx->add_module(std::move(module));
    auto kernel_symbol = tlctx->lookup_function_pointer(offloaded_task_name);
    return [=](Context &context) {
      TI_TRACE("Launching Taichi Kernel Function");
      auto func = (int32 (*)(void *))kernel_symbol;
      func(&context);
    };
  }
};

FunctionType CodeGenWASM::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMWASM(kernel, ir).gen();
}

}  // namespace lang
}  // namespace taichi
