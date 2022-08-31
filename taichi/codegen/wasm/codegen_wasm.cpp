#include "taichi/codegen/wasm/codegen_wasm.h"

#include "taichi/codegen/llvm/codegen_llvm.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/statistics.h"
#include "taichi/util/file_sequence_writer.h"

namespace taichi {
namespace lang {

namespace {
constexpr std::array<const char *, 5> kPreloadedFuncNames = {
    "wasm_materialize", "wasm_set_kernel_parameter_i32",
    "wasm_set_kernel_parameter_f32", "wasm_set_print_buffer", "wasm_print"};
}

class TaskCodeGenWASM : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenWASM(Kernel *kernel,
                  IRNode *ir,
                  std::unique_ptr<llvm::Module> &&M = nullptr)
      : TaskCodeGenLLVM(kernel, ir, std::move(M)) {
    TI_AUTO_PROF
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    [[maybe_unused]] int step = 1;

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
#ifdef TI_LLVM_15
      auto *loop_var_load = builder->CreateLoad(begin->getType(), loop_var);
#else
      auto *loop_var_load = builder->CreateLoad(loop_var);
#endif
      if (!stmt->reversed) {
        cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                   loop_var_load, end);
      } else {
        cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE,
                                   loop_var_load, begin);
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

  void visit(PrintStmt *stmt) override {
    std::vector<llvm::Value *> args;
    for (auto const &content : stmt->contents) {
      if (std::holds_alternative<Stmt *>(content)) {
        auto arg_stmt = std::get<Stmt *>(content);
        auto value = llvm_val[arg_stmt];
        if (arg_stmt->ret_type->is_primitive(PrimitiveTypeID::i32)) {
          auto func = get_runtime_function("wasm_print_i32");
          builder->CreateCall(func,
                              std::vector<llvm::Value *>{get_context(), value});
        } else if (arg_stmt->ret_type->is_primitive(PrimitiveTypeID::f32)) {
          auto func = get_runtime_function("wasm_print_f32");
          builder->CreateCall(func,
                              std::vector<llvm::Value *>{get_context(), value});
        } else {
          TI_NOT_IMPLEMENTED
        }
      } else {
        auto arg_str = std::get<std::string>(content);
        for (int i = 0; i < (int)arg_str.size(); i += 4) {
          llvm::Value *values[4];
          for (int j = 0; j < 4; ++j)
            if (i + j < (int)arg_str.size()) {
              values[j] = llvm::ConstantInt::get(
                  *llvm_context, llvm::APInt(8, (uint64)arg_str[i + j], true));
            } else {
              values[j] = llvm::ConstantInt::get(
                  *llvm_context, llvm::APInt(8, (uint64)0, true));
            }
          auto func = get_runtime_function("wasm_print_char");
          builder->CreateCall(func, std::vector<llvm::Value *>{
                                        get_context(), values[0], values[1],
                                        values[2], values[3]});
        }
      }
    }
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

  LLVMCompiledData run_compilation() override {
    // lower kernel
    if (!kernel->lowered()) {
      kernel->lower();
    }
    // emit_to_module
    stat.add("codegen_taichi_kernel_function");
    auto offloaded_task_name = init_taichi_kernel_function();
    ir->accept(this);
    finalize_taichi_kernel_function();
    // only keep the current func
    TaichiLLVMContext::eliminate_unused_functions(
        module.get(), [offloaded_task_name](const std::string &func_name) {
          for (auto &name : kPreloadedFuncNames) {
            if (std::string(name) == func_name) {
              return true;
            }
          }
          return func_name == offloaded_task_name;
        });
    LLVMCompiledData res;
    res.tasks.emplace_back(offloaded_task_name);
    res.module = std::move(this->module);
    return res;
  }
};

FunctionType KernelCodeGenWASM::compile_to_function() {
  TI_AUTO_PROF
  TaskCodeGenWASM gen(kernel, ir);
  auto res = gen.run_compilation();
  gen.tlctx->add_module(std::move(res.module));
  auto kernel_symbol = gen.tlctx->lookup_function_pointer(res.tasks[0].name);
  return [=](RuntimeContext &context) {
    TI_TRACE("Launching Taichi Kernel Function");
    auto func = (int32(*)(void *))kernel_symbol;
    func(&context);
  };
}

LLVMCompiledData KernelCodeGenWASM::compile_task(
    std::unique_ptr<llvm::Module> &&module,
    OffloadedStmt *stmt) {
  bool init_flag = module == nullptr;
  std::vector<OffloadedTask> name_list;
  auto gen = std::make_unique<TaskCodeGenWASM>(kernel, ir, std::move(module));

  name_list.emplace_back(nullptr);
  name_list[0].name = gen->init_taichi_kernel_function();
  gen->emit_to_module();
  gen->finalize_taichi_kernel_function();

  // TODO: move the following functions to dump process in AOT.
  if (init_flag) {
    for (auto &name : kPreloadedFuncNames) {
      name_list.emplace_back(nullptr);
      name_list.back().name = name;
    }
  }

  gen->tlctx->jit->global_optimize_module(gen->module.get());

  return {name_list, std::move(gen->module), {}, {}};
}
}  // namespace lang
}  // namespace taichi
