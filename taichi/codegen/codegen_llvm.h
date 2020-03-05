// The LLVM backend
#pragma once

#include <set>
#include <taichi/common/util.h>
#include <taichi/util/io.h>

#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/lang_util.h"

#include "taichi/llvm/llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

class CodeGenLLVM : public IRVisitor, public ModuleBuilder {
 public:
  Kernel *kernel;
  Program *prog;
  std::string kernel_name;
  std::vector<Value *> kernel_args;
  llvm::Type *context_ty;
  llvm::Type *physical_coordinate_ty;

  llvm::Value *current_coordinates;
  llvm::BasicBlock *while_after_loop;
  llvm::FunctionType *task_function_type;
  OffloadedStmt *current_offloaded_stmt;
  SNodeAttributes &snode_attr;
  int task_counter;

  using IRVisitor::visit;
  using ModuleBuilder::call;

  Arch current_arch() {
    return kernel->arch;
  }

  void initialize_context() {
    if (kernel->arch == Arch::cuda) {
      tlctx = prog->llvm_context_device.get();
    } else {
      tlctx = prog->llvm_context_host.get();
    }
    llvm_context = tlctx->ctx.get();
    builder = std::make_unique<llvm::IRBuilder<>>(*llvm_context);
  }

  llvm::Function *func;

  class OffloadedTask {
   public:
    std::string name;
    CodeGenLLVM *codegen;
    using task_fp_type = int32 (*)(void *);
    task_fp_type func;

    int block_dim;
    int grid_dim;
    void *cuda_func;

    OffloadedTask(CodeGenLLVM *codegen) : codegen(codegen) {
      func = nullptr;
    }

    void begin(const std::string &name) {
      this->name = name;
    }

    void end() {
      codegen->offloaded_tasks.push_back(*this);
    }

    void compile() {
      TI_ASSERT(!func);
      auto kernel_symbol = codegen->tlctx->lookup_function_pointer(name);
      TI_ASSERT_INFO(kernel_symbol, "Function not found");

      func = (task_fp_type)kernel_symbol;
    }

    void operator()(Context *context) {
      TI_ASSERT(func);
      func(context);
    }
  };

  std::unique_ptr<OffloadedTask> current_task;
  std::vector<OffloadedTask> offloaded_tasks;

  CodeGenLLVM(Kernel *kernel)
      // TODO: simplify ModuleBuilder ctor input
      : ModuleBuilder(kernel->program.get_llvm_context(kernel->arch)
                          ->clone_struct_module()),
        kernel(kernel),
        prog(&kernel->program),
        snode_attr(prog->get_llvm_context(kernel->arch)->snode_attr),
        task_counter(0) {
    initialize_context();

    context_ty = get_runtime_type("Context");
    physical_coordinate_ty = get_runtime_type("PhysicalCoordinates");

    std::string grad_suffix;
    if (kernel->grad) {
      grad_suffix = "_grad";
    }
    kernel_name = kernel->name + grad_suffix + "_kernel";
  }

  llvm::Value *get_arg(int i) {
    std::vector<llvm::Value *> args;
    for (auto &arg : func->args()) {
      args.push_back(&arg);
    }
    return args[i];
  }

  llvm::Value *get_context() {
    return get_arg(0);
  }

  llvm::Value *get_root() {
    return create_call("LLVMRuntime_get_root", {get_runtime()});
  }

  llvm::Value *get_runtime() {
    auto runtime_ptr = create_call("Context_get_runtime", {get_context()});
    return builder->CreateBitCast(
        runtime_ptr,
        llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0));
  }

  void emit_struct_meta_base(const std::string &name,
                             llvm::Value *node_meta,
                             SNode *snode) {
    RuntimeObject common("StructMeta", this, builder.get(), node_meta);
    std::size_t element_size;
    if (snode->type == SNodeType::dense) {
      auto element_ty = snode_attr[snode].llvm_body_type->getArrayElementType();
      element_size = tlctx->get_type_size(element_ty);
    } else if (snode->type == SNodeType::pointer) {
      auto element_ty = tlctx->snode_attr[snode->ch[0]].llvm_type;
      element_size = tlctx->get_type_size(element_ty);
    } else {
      auto element_ty = tlctx->snode_attr[snode].llvm_element_type;
      element_size = tlctx->get_type_size(element_ty);
    }
    common.set("snode_id", tlctx->get_constant(snode->id));
    common.set("element_size", tlctx->get_constant((uint64)element_size));
    common.set("max_num_elements",
               tlctx->get_constant(1 << snode->total_num_bits));
    common.set("context", get_context());

    /*
    uint8 *(*lookup_element)(uint8 *, int i);
    uint8 *(*from_parent_element)(uint8 *);
    bool (*is_active)(uint8 *, int i);
    int (*get_num_elements)(uint8 *);
    void (*refine_coordinates)(PhysicalCoordinates *inp_coord,
                               PhysicalCoordinates *refined_coord,
                               int index);
                               */

    std::vector<std::string> functions = {"lookup_element", "is_active",
                                          "get_num_elements"};

    for (auto const f : functions)
      common.set(f, get_runtime_function(fmt::format("{}_{}", name, f)));

    // "from_parent_element", "refine_coordinates" are different for different
    // snodes, even if they have the same type.
    if (snode->parent)
      common.set("from_parent_element",
                 get_runtime_function(snode->get_ch_from_parent_func_name()));

    if (snode->type != SNodeType::place)
      common.set("refine_coordinates",
                 get_runtime_function(snode->refine_coordinates_func_name()));
  }

  std::unique_ptr<RuntimeObject> emit_struct_meta_object(SNode *snode) {
    std::unique_ptr<RuntimeObject> meta;
    if (snode->type == SNodeType::dense) {
      meta = std::make_unique<RuntimeObject>("DenseMeta", this, builder.get());
      emit_struct_meta_base("Dense", meta->ptr, snode);
      meta->call("set_bitmasked", tlctx->get_constant(snode->_bitmasked));
      meta->call("set_morton_dim", tlctx->get_constant((int)snode->_morton));
    } else if (snode->type == SNodeType::pointer) {
      meta =
          std::make_unique<RuntimeObject>("pointerMeta", this, builder.get());
      emit_struct_meta_base("pointer", meta->ptr, snode);
    } else if (snode->type == SNodeType::root) {
      meta = std::make_unique<RuntimeObject>("RootMeta", this, builder.get());
      emit_struct_meta_base("Root", meta->ptr, snode);
    } else if (snode->type == SNodeType::dynamic) {
      meta =
          std::make_unique<RuntimeObject>("DynamicMeta", this, builder.get());
      emit_struct_meta_base("Dynamic", meta->ptr, snode);
      meta->call("set_chunk_size", tlctx->get_constant(snode->chunk_size));
    } else {
      TI_P(snode_type_name(snode->type));
      TI_NOT_IMPLEMENTED;
    }
    if (false) {
      // auto ptr_type = llvm::Type::getInt8PtrTy(*llvm_context, 0);
      auto ptr_type = llvm::PointerType::get(meta->type, 0);
      auto ptr = meta->ptr;  // builder->CreatePointerCast(meta->ptr, ptr_type);
      auto struct_meta_size = tlctx->get_type_size(meta->type);
      builder->CreateIntrinsic(llvm::Intrinsic::invariant_start, {ptr_type},
                               {tlctx->get_constant(struct_meta_size), ptr});
    }
    return meta;
  }

  llvm::Value *emit_struct_meta(SNode *snode) {
    auto obj = emit_struct_meta_object(snode);
    TI_ASSERT(obj != nullptr);
    return obj->ptr;
  }

  virtual void emit_to_module() {
    kernel->ir->accept(this);
  }

  virtual FunctionType compile_module_to_executable() {
    tlctx->add_module(std::move(module));

    for (auto &task : offloaded_tasks) {
      task.compile();
    }
    auto offloaded_tasks_local = offloaded_tasks;
    return [=](Context &context) {
      for (auto task : offloaded_tasks_local) {
        task(&context);
      }
    };
  }

  virtual FunctionType gen() {
    emit_to_module();
    return compile_module_to_executable();
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    TI_NOT_IMPLEMENTED
  }

  void visit(Block *stmt_list) override {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    stmt->value = create_entry_block_alloca(stmt->ret_type.data_type);
    // initialize as zero
    builder->CreateStore(tlctx->get_constant(stmt->ret_type.data_type, 0),
                         stmt->value);
  }

  void visit(RandStmt *stmt) override {
    stmt->value = create_call(
        fmt::format("rand_{}", data_type_short_name(stmt->ret_type.data_type)));
  }

  virtual void emit_extra_unary(UnaryOpStmt *stmt) {
    auto input = stmt->operand->value;
    auto input_taichi_type = stmt->operand->ret_type.data_type;
    auto op = stmt->op_type;
    auto input_type = input->getType();

#define UNARY_STD(x)                                                   \
  else if (op == UnaryOpType::x) {                                     \
    if (input_taichi_type == DataType::f32) {                          \
      stmt->value =                                                    \
          builder->CreateCall(get_runtime_function(#x "_f32"), input); \
    } else if (input_taichi_type == DataType::f64) {                   \
      stmt->value =                                                    \
          builder->CreateCall(get_runtime_function(#x "_f64"), input); \
    } else if (input_taichi_type == DataType::i32) {                   \
      stmt->value =                                                    \
          builder->CreateCall(get_runtime_function(#x "_i32"), input); \
    } else {                                                           \
      TI_NOT_IMPLEMENTED                                               \
    }                                                                  \
  }
    if (false) {
    }
    UNARY_STD(abs)
    UNARY_STD(exp)
    UNARY_STD(log)
    UNARY_STD(tan)
    UNARY_STD(tanh)
    UNARY_STD(sgn)
    UNARY_STD(logic_not)
    UNARY_STD(acos)
    UNARY_STD(asin)
    UNARY_STD(cos)
    UNARY_STD(sin)
    else if (op == UnaryOpType::sqrt) {
      stmt->value = builder->CreateIntrinsic(llvm::Intrinsic::sqrt,
                                             {input_type}, {input});
    }
    else {
      TI_P(unary_op_type_name(op));
      TI_NOT_IMPLEMENTED
    }
#undef UNARY_STD
  }

  void visit(UnaryOpStmt *stmt) override {
    auto input = stmt->operand->value;
    auto input_type = input->getType();
    auto op = stmt->op_type;

#define UNARY_INTRINSIC(x)                                                   \
  else if (op == UnaryOpType::x) {                                           \
    stmt->value =                                                            \
        builder->CreateIntrinsic(llvm::Intrinsic::x, {input_type}, {input}); \
  }

    if (stmt->op_type != UnaryOpType::cast) {
      if (op == UnaryOpType::rsqrt) {
        llvm::Function *sqrt_fn = Intrinsic::getDeclaration(
            module.get(), Intrinsic::sqrt, input->getType());
        auto intermediate = builder->CreateCall(sqrt_fn, input, "sqrt");
        stmt->value = builder->CreateFDiv(
            tlctx->get_constant(stmt->ret_type.data_type, 1.0), intermediate);
      } else if (op == UnaryOpType::bit_not) {
        stmt->value = builder->CreateNot(input);
      } else if (op == UnaryOpType::neg) {
        if (is_real(stmt->operand->ret_type.data_type)) {
          stmt->value = builder->CreateFNeg(input, "neg");
        } else {
          stmt->value = builder->CreateNeg(input, "neg");
        }
      }
      UNARY_INTRINSIC(floor)
      UNARY_INTRINSIC(ceil)
      else emit_extra_unary(stmt);
#undef UNARY_INTRINSIC
    } else {
      // op = cast
      if (stmt->cast_by_value) {
        llvm::CastInst::CastOps cast_op;
        auto from = stmt->operand->ret_type.data_type;
        auto to = stmt->cast_type;
        TI_ASSERT(from != to);
        if (is_real(from) != is_real(to)) {
          if (is_real(from) && is_integral(to)) {
            cast_op = llvm::Instruction::CastOps::FPToSI;
          } else if (is_integral(from) && is_real(to)) {
            cast_op = llvm::Instruction::CastOps::SIToFP;
          } else {
            TI_P(data_type_name(from));
            TI_P(data_type_name(to));
            TI_NOT_IMPLEMENTED;
          }
          stmt->value =
              builder->CreateCast(cast_op, stmt->operand->value,
                                  tlctx->get_data_type(stmt->cast_type));
        } else if (is_real(from) && is_real(to)) {
          if (data_type_size(from) < data_type_size(to)) {
            stmt->value = builder->CreateFPExt(
                stmt->operand->value, tlctx->get_data_type(stmt->cast_type));
          } else {
            stmt->value = builder->CreateFPTrunc(
                stmt->operand->value, tlctx->get_data_type(stmt->cast_type));
          }
        } else if (!is_real(from) && !is_real(to)) {
          if (data_type_size(from) < data_type_size(to)) {
            stmt->value = builder->CreateSExt(
                stmt->operand->value, tlctx->get_data_type(stmt->cast_type));
          } else {
            stmt->value = builder->CreateTrunc(
                stmt->operand->value, tlctx->get_data_type(stmt->cast_type));
          }
        }
      } else {
        TI_ASSERT(data_type_size(stmt->ret_type.data_type) ==
                  data_type_size(stmt->cast_type));
        stmt->value = builder->CreateBitCast(
            stmt->operand->value, tlctx->get_data_type(stmt->cast_type));
      }
    }
  }

  llvm::Type *llvm_type(DataType dt) {
    if (dt == DataType::i32) {
      return llvm::Type::getInt32Ty(*llvm_context);
    } else if (dt == DataType::i1) {
      return llvm::Type::getInt1Ty(*llvm_context);
    } else if (dt == DataType::f32) {
      return llvm::Type::getFloatTy(*llvm_context);
    } else if (dt == DataType::f64) {
      return llvm::Type::getDoubleTy(*llvm_context);
    } else {
      TI_NOT_IMPLEMENTED;
    }
    return nullptr;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto op = stmt->op_type;
    auto ret_type = stmt->ret_type.data_type;
    if (op == BinaryOpType::add) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder->CreateFAdd(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder->CreateAdd(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::sub) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder->CreateFSub(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder->CreateSub(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::mul) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder->CreateFMul(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder->CreateMul(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::floordiv) {
      if (is_integral(ret_type))
        stmt->value = create_call(
            fmt::format("floordiv_{}", data_type_short_name(ret_type)),
            {stmt->lhs->value, stmt->rhs->value});
      else {
        auto div = builder->CreateFDiv(stmt->lhs->value, stmt->rhs->value);
        stmt->value = builder->CreateIntrinsic(
            llvm::Intrinsic::floor, {tlctx->get_data_type(ret_type)}, {div});
      }
    } else if (op == BinaryOpType::div) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder->CreateFDiv(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder->CreateSDiv(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::mod) {
      stmt->value = builder->CreateSRem(stmt->lhs->value, stmt->rhs->value);
    } else if (op == BinaryOpType::bit_and) {
      stmt->value = builder->CreateAnd(stmt->lhs->value, stmt->rhs->value);
    } else if (op == BinaryOpType::bit_or) {
      stmt->value = builder->CreateOr(stmt->lhs->value, stmt->rhs->value);
    } else if (op == BinaryOpType::bit_xor) {
      stmt->value = builder->CreateXor(stmt->lhs->value, stmt->rhs->value);
    } else if (op == BinaryOpType::max) {
      if (is_real(ret_type)) {
        stmt->value = builder->CreateMaxNum(stmt->lhs->value, stmt->rhs->value);
      } else if (ret_type == DataType::i32) {
        stmt->value =
            create_call("max_i32", {stmt->lhs->value, stmt->rhs->value});
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::atan2) {
      if (arch_is_cpu(current_arch())) {
        if (ret_type == DataType::f32) {
          stmt->value =
              create_call("atan2_f32", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::f64) {
          stmt->value =
              create_call("atan2_f64", {stmt->lhs->value, stmt->rhs->value});
        } else {
          TI_P(data_type_name(ret_type));
          TI_NOT_IMPLEMENTED
        }
      } else if (current_arch() == Arch::cuda) {
        if (ret_type == DataType::f32) {
          stmt->value =
              create_call("__nv_atan2f", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::f64) {
          stmt->value =
              create_call("__nv_atan2", {stmt->lhs->value, stmt->rhs->value});
        } else {
          TI_P(data_type_name(ret_type));
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::pow) {
      if (arch_is_cpu(current_arch())) {
        if (ret_type == DataType::f32) {
          stmt->value =
              create_call("pow_f32", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::f64) {
          stmt->value =
              create_call("pow_f64", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::i32) {
          stmt->value =
              create_call("pow_i32", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::i64) {
          stmt->value =
              create_call("pow_i64", {stmt->lhs->value, stmt->rhs->value});
        } else {
          TI_P(data_type_name(ret_type));
          TI_NOT_IMPLEMENTED
        }
      } else if (current_arch() == Arch::cuda) {
        if (ret_type == DataType::f32) {
          stmt->value =
              create_call("__nv_powf", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::f64) {
          stmt->value =
              create_call("__nv_pow", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::i32) {
          stmt->value =
              create_call("pow_i32", {stmt->lhs->value, stmt->rhs->value});
        } else if (ret_type == DataType::i64) {
          stmt->value =
              create_call("pow_i64", {stmt->lhs->value, stmt->rhs->value});
        } else {
          TI_P(data_type_name(ret_type));
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::min) {
      if (is_real(ret_type)) {
        stmt->value = builder->CreateMinNum(stmt->lhs->value, stmt->rhs->value);
      } else if (ret_type == DataType::i32) {
        stmt->value =
            create_call("min_i32", {stmt->lhs->value, stmt->rhs->value});
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else if (is_comparison(op)) {
      llvm::Value *cmp = nullptr;
      auto input_type = stmt->lhs->ret_type.data_type;
      if (op == BinaryOpType::cmp_eq) {
        if (is_real(input_type)) {
          cmp = builder->CreateFCmpOEQ(stmt->lhs->value, stmt->rhs->value);
        } else {
          cmp = builder->CreateICmpEQ(stmt->lhs->value, stmt->rhs->value);
        }
      } else if (op == BinaryOpType::cmp_le) {
        if (is_real(input_type)) {
          cmp = builder->CreateFCmpOLE(stmt->lhs->value, stmt->rhs->value);
        } else {
          if (is_signed(input_type)) {
            cmp = builder->CreateICmpSLE(stmt->lhs->value, stmt->rhs->value);
          } else {
            cmp = builder->CreateICmpULE(stmt->lhs->value, stmt->rhs->value);
          }
        }
      } else if (op == BinaryOpType::cmp_ge) {
        if (is_real(input_type)) {
          cmp = builder->CreateFCmpOGE(stmt->lhs->value, stmt->rhs->value);
        } else {
          if (is_signed(input_type)) {
            cmp = builder->CreateICmpSGE(stmt->lhs->value, stmt->rhs->value);
          } else {
            cmp = builder->CreateICmpUGE(stmt->lhs->value, stmt->rhs->value);
          }
        }
      } else if (op == BinaryOpType::cmp_lt) {
        if (is_real(input_type)) {
          cmp = builder->CreateFCmpOLT(stmt->lhs->value, stmt->rhs->value);
        } else {
          if (is_signed(input_type)) {
            cmp = builder->CreateICmpSLT(stmt->lhs->value, stmt->rhs->value);
          } else {
            cmp = builder->CreateICmpULT(stmt->lhs->value, stmt->rhs->value);
          }
        }
      } else if (op == BinaryOpType::cmp_gt) {
        if (is_real(input_type)) {
          cmp = builder->CreateFCmpOGT(stmt->lhs->value, stmt->rhs->value);
        } else {
          if (is_signed(input_type)) {
            cmp = builder->CreateICmpSGT(stmt->lhs->value, stmt->rhs->value);
          } else {
            cmp = builder->CreateICmpUGT(stmt->lhs->value, stmt->rhs->value);
          }
        }
      } else if (op == BinaryOpType::cmp_ne) {
        if (is_real(input_type)) {
          cmp = builder->CreateFCmpONE(stmt->lhs->value, stmt->rhs->value);
        } else {
          cmp = builder->CreateICmpNE(stmt->lhs->value, stmt->rhs->value);
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
      stmt->value = builder->CreateSExt(cmp, llvm_type(DataType::i32));
    } else {
      TI_P(binary_op_type_name(op));
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(TernaryOpStmt *stmt) override {
    TI_ASSERT(stmt->op_type == TernaryOpType::select);
    stmt->value = builder->CreateSelect(
        builder->CreateTrunc(stmt->op1->value, llvm_type(DataType::i1)),
        stmt->op2->value, stmt->op3->value);
  }

  void visit(IfStmt *if_stmt) override {
    // TODO: take care of vectorized cases
    BasicBlock *true_block =
        BasicBlock::Create(*llvm_context, "true_block", func);
    BasicBlock *false_block =
        BasicBlock::Create(*llvm_context, "false_block", func);
    BasicBlock *after_if = BasicBlock::Create(*llvm_context, "after_if", func);
    builder->CreateCondBr(
        builder->CreateICmpNE(if_stmt->cond->value, tlctx->get_constant(0)),
        true_block, false_block);
    builder->SetInsertPoint(true_block);
    if (if_stmt->true_statements) {
      if_stmt->true_statements->accept(this);
    }
    builder->CreateBr(after_if);
    builder->SetInsertPoint(false_block);
    if (if_stmt->false_statements) {
      if_stmt->false_statements->accept(this);
    }
    builder->CreateBr(after_if);
    builder->SetInsertPoint(after_if);
  }

  llvm::Value *create_print(std::string tag, DataType dt, llvm::Value *value) {
    std::vector<Value *> args;
    std::string format;
    if (dt == DataType::i32) {
      format = "%d";
    } else if (dt == DataType::i64) {
#if defined(TI_PLATFORM_UNIX)
      format = "%lld";
#else
      format = "%I64d";
#endif
    } else if (dt == DataType::f32) {
      format = "%f";
      value = builder->CreateFPExt(value, tlctx->get_data_type(DataType::f64));
    } else if (dt == DataType::f64) {
      format = "%.12f";
    } else {
      TI_NOT_IMPLEMENTED
    }
    args.push_back(builder->CreateGlobalStringPtr(
        ("[llvm codegen debug] " + tag + " = " + format + "\n").c_str(),
        "format_string"));
    args.push_back(value);
    return builder->CreateCall(get_runtime_function("printf"), args,
                               "debug_printf");
  }

  void visit(PrintStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    std::vector<Value *> args;
    std::string format;
    auto value = stmt->stmt->value;
    auto dt = stmt->stmt->ret_type.data_type;
    if (dt == DataType::i32) {
      format = "%d";
    } else if (dt == DataType::i64) {
#if defined(TI_PLATFORM_UNIX)
      format = "%lld";
#else
      format = "%I64d";
#endif
    } else if (dt == DataType::f32) {
      format = "%f";
      value = builder->CreateFPExt(value, tlctx->get_data_type(DataType::f64));
    } else if (dt == DataType::f64) {
      format = "%.12f";
    } else {
      TI_NOT_IMPLEMENTED
    }
    auto runtime_printf = call("LLVMRuntime_get_host_printf", get_runtime());
    args.push_back(builder->CreateGlobalStringPtr(
        ("[debug] " + stmt->str + " = " + format + "\n").c_str(),
        "format_string"));
    args.push_back(value);

    stmt->value = builder->CreateCall(runtime_printf, args);
  }

  void visit(ConstStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    auto val = stmt->val[0];
    if (val.dt == DataType::f32) {
      stmt->value = llvm::ConstantFP::get(*llvm_context,
                                          llvm::APFloat(val.val_float32()));
    } else if (val.dt == DataType::f64) {
      stmt->value = llvm::ConstantFP::get(*llvm_context,
                                          llvm::APFloat(val.val_float64()));
    } else if (val.dt == DataType::i32) {
      stmt->value = llvm::ConstantInt::get(
          *llvm_context, llvm::APInt(32, val.val_int32(), true));
    } else if (val.dt == DataType::i64) {
      stmt->value = llvm::ConstantInt::get(
          *llvm_context, llvm::APInt(64, val.val_int64(), true));
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }

  void visit(WhileControlStmt *stmt) override {
    BasicBlock *after_break =
        BasicBlock::Create(*llvm_context, "after_break", func);
    TI_ASSERT(while_after_loop);
    auto cond =
        builder->CreateICmpEQ(stmt->cond->value, tlctx->get_constant(0));
    builder->CreateCondBr(cond, while_after_loop, after_break);
    builder->SetInsertPoint(after_break);
  }

  void visit(WhileStmt *stmt) override {
    BasicBlock *body =
        BasicBlock::Create(*llvm_context, "while_loop_body", func);
    builder->CreateBr(body);
    builder->SetInsertPoint(body);

    BasicBlock *after_loop =
        BasicBlock::Create(*llvm_context, "after_while", func);
    auto old_while_after_loop = while_after_loop;
    while_after_loop = after_loop;

    stmt->body->accept(this);

    builder->CreateBr(body);  // jump to head

    builder->SetInsertPoint(after_loop);
    while_after_loop = old_while_after_loop;
  }

  llvm::Value *cast_pointer(llvm::Value *val,
                            std::string dest_ty_name,
                            int addr_space = 0) {
    return builder->CreateBitCast(
        val,
        llvm::PointerType::get(get_runtime_type(dest_ty_name), addr_space));
  }

  void emit_clear_list(OffloadedStmt *listgen) {
    auto snode_child = listgen->snode;
    auto snode_parent = listgen->snode->parent;
    auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
    auto meta_parent =
        cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
    call("clear_list", get_runtime(), meta_parent, meta_child);
  }

  void emit_list_gen(OffloadedStmt *listgen) {
    auto snode_child = listgen->snode;
    auto snode_parent = listgen->snode->parent;
    auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
    auto meta_parent =
        cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
    call("element_listgen", get_runtime(), meta_parent, meta_child);
  }

  void emit_gc(OffloadedStmt *stmt) {
    auto snode = stmt->snode->id;
    call("node_gc", get_runtime(), tlctx->get_constant(snode));
  }

  llvm::Value *create_call(llvm::Value *func, std::vector<Value *> args = {}) {
    check_func_call_signature(func, args);
    return builder->CreateCall(func, args);
  }

  llvm::Value *create_call(std::string func_name,
                           std::vector<Value *> args = {}) {
    auto func = get_runtime_function(func_name);
    return create_call(func, args);
  }

  void create_increment(llvm::Value *ptr, llvm::Value *value) {
    builder->CreateStore(builder->CreateAdd(builder->CreateLoad(ptr), value),
                         ptr);
  }

  // Direct translation
  void create_naive_range_for(RangeForStmt *for_stmt) {
    BasicBlock *body = BasicBlock::Create(*llvm_context, "loop_body", func);
    BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "block", func);
    BasicBlock *test = BasicBlock::Create(*llvm_context, "test", func);
    if (!for_stmt->reversed) {
      builder->CreateStore(for_stmt->begin->value, for_stmt->loop_var->value);
    } else {
      builder->CreateStore(
          builder->CreateSub(for_stmt->end->value, tlctx->get_constant(1)),
          for_stmt->loop_var->value);
    }
    builder->CreateBr(test);

    {
      // test block
      builder->SetInsertPoint(test);
      llvm::Value *cond;
      if (!for_stmt->reversed) {
        cond =
            builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                builder->CreateLoad(for_stmt->loop_var->value),
                                for_stmt->end->value);
      } else {
        cond =
            builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE,
                                builder->CreateLoad(for_stmt->loop_var->value),
                                for_stmt->begin->value);
      }
      builder->CreateCondBr(cond, body, after_loop);
    }

    {
      // body cfg
      builder->SetInsertPoint(body);

      for_stmt->body->accept(this);

      if (!for_stmt->reversed) {
        create_increment(for_stmt->loop_var->value, tlctx->get_constant(1));
      } else {
        create_increment(for_stmt->loop_var->value, tlctx->get_constant(-1));
      }
      builder->CreateBr(test);
    }

    // next cfg
    builder->SetInsertPoint(after_loop);
  }

  virtual void visit(RangeForStmt *for_stmt) override {
    create_naive_range_for(for_stmt);
  }

  void visit(ArgLoadStmt *stmt) override {
    auto raw_arg = call(builder.get(), "Context_get_args", get_context(),
                        tlctx->get_constant(stmt->arg_id));

    llvm::Type *dest_ty = nullptr;
    if (stmt->is_ptr) {
      dest_ty = PointerType::get(tlctx->get_data_type(DataType::i32), 0);
      stmt->value = builder->CreateIntToPtr(raw_arg, dest_ty);
    } else {
      dest_ty = tlctx->get_data_type(stmt->ret_type.data_type);
      auto dest_bits = dest_ty->getPrimitiveSizeInBits();
      auto truncated = builder->CreateTrunc(
          raw_arg, Type::getIntNTy(*llvm_context, dest_bits));
      stmt->value = builder->CreateBitCast(truncated, dest_ty);
    }
  }

  void visit(ArgStoreStmt *stmt) override {
    if (stmt->is_ptr) {
      TI_NOT_IMPLEMENTED
    } else {
      auto intermediate_bits =
          tlctx->get_data_type(stmt->val->ret_type.data_type)
              ->getPrimitiveSizeInBits();
      llvm::Type *intermediate_type =
          llvm::Type::getIntNTy(*llvm_context, intermediate_bits);
      llvm::Type *dest_ty = tlctx->get_data_type<int64>();
      auto extended = builder->CreateZExt(
          builder->CreateBitCast(stmt->val->value, intermediate_type), dest_ty);
      // TODO: refactor this part
      if (get_current_program().config.arch == Arch::cuda &&
          !get_current_program().config.use_unified_memory) {
        // For SNode reader without unified memory. This is a temporary
        // solution.
        builder->CreateCall(get_runtime_function("LLVMRuntime_store_result"),
                            {get_runtime(), extended});
      } else {
        builder->CreateCall(
            get_runtime_function("Context_set_args"),
            {get_context(), tlctx->get_constant(stmt->arg_id), extended});
      }
    }
  }

  void visit(LocalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    stmt->value = builder->CreateLoad(stmt->ptr[0].var->value);
  }

  void visit(LocalStoreStmt *stmt) override {
    auto mask = stmt->parent->mask();
    if (mask && stmt->width() != 1) {
      TI_NOT_IMPLEMENTED
    } else {
      builder->CreateStore(stmt->data->value, stmt->ptr->value);
    }
  }

  void visit(AssertStmt *stmt) override {
    stmt->value = call("taichi_assert", get_context(), stmt->val->value,
                       builder->CreateGlobalStringPtr(stmt->text));
  }

  void visit(SNodeOpStmt *stmt) override {
    auto snode = stmt->snode;
    if (stmt->op_type == SNodeOpType::append) {
      TI_ASSERT(snode->type == SNodeType::dynamic);
      TI_ASSERT(stmt->ret_type.data_type == DataType::i32);
      stmt->value = call(snode, stmt->ptr->value, "append", {stmt->val->value});
    } else if (stmt->op_type == SNodeOpType::length) {
      TI_ASSERT(snode->type == SNodeType::dynamic);
      stmt->value = call(snode, stmt->ptr->value, "get_num_elements", {});
    } else if (stmt->op_type == SNodeOpType::is_active) {
      stmt->value =
          call(snode, stmt->ptr->value, "is_active", {stmt->val->value});
    } else if (stmt->op_type == SNodeOpType::deactivate) {
      if (snode->type == SNodeType::pointer || snode->type == SNodeType::hash) {
        stmt->value =
            call(snode, stmt->ptr->value, "deactivate", {stmt->val->value});
      } else if (snode->type == SNodeType::dynamic) {
        stmt->value = call(snode, stmt->ptr->value, "deactivate", {});
      }
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    // auto mask = stmt->parent->mask();
    // TODO: deal with mask when vectorized
    TI_ASSERT(stmt->width() == 1);
    for (int l = 0; l < stmt->width(); l++) {
      TI_ASSERT(stmt->op_type == AtomicOpType::add);
      llvm::Value *old_value;
      if (stmt->val->ret_type.data_type == DataType::i32)
        old_value = builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::BinOp::Add, stmt->dest->value,
            stmt->val->value, llvm::AtomicOrdering::SequentiallyConsistent);
      else if (stmt->val->ret_type.data_type == DataType::f32) {
        old_value = builder->CreateCall(get_runtime_function("atomic_add_f32"),
                                        {stmt->dest->value, stmt->val->value});
      } else if (stmt->val->ret_type.data_type == DataType::f64) {
        old_value = builder->CreateCall(get_runtime_function("atomic_add_f64"),
                                        {stmt->dest->value, stmt->val->value});
      } else {
        TI_NOT_IMPLEMENTED
      }
      stmt->value = old_value;
    }
  }

  void visit(GlobalPtrStmt *stmt) override {
    TI_ERROR("Global Ptrs should have been lowered.");
  }

  void visit(GlobalStoreStmt *stmt) override {
    TI_ASSERT(!stmt->parent->mask() || stmt->width() == 1);
    TI_ASSERT(stmt->data->value);
    TI_ASSERT(stmt->ptr->value);
    builder->CreateStore(stmt->data->value, stmt->ptr->value);
  }

  void visit(GlobalLoadStmt *stmt) override {
    int width = stmt->width();
    TI_ASSERT(width == 1);
    stmt->value = builder->CreateLoad(
        tlctx->get_data_type(stmt->ret_type.data_type), stmt->ptr->value);
  }

  void visit(ElementShuffleStmt *stmt) override {
    auto init = stmt->elements.serialize(
        [](const VectorElement &elem) {
          return fmt::format("{}[{}]", elem.stmt->raw_name(), elem.index);
        },
        "{");
    if (stmt->pointer) {
      emit("{} * const {} [{}] {};", data_type_name(stmt->ret_type.data_type),
           stmt->raw_name(), stmt->width(), init);
    } else {
      emit("const {} {} ({});", stmt->ret_data_type_name(), stmt->raw_name(),
           init);
    }
  }

  static std::string get_runtime_snode_name(SNode *snode) {
    if (snode->type == SNodeType::root) {
      return "Root";
    } else if (snode->type == SNodeType::dense) {
      return "Dense";
    } else if (snode->type == SNodeType::dynamic) {
      return "Dynamic";
    } else if (snode->type == SNodeType::pointer) {
      return "pointer";
    } else if (snode->type == SNodeType::hash) {
      return "Hash";
    } else {
      TI_P(snode_type_name(snode->type));
      TI_NOT_IMPLEMENTED
    }
  }

  llvm::Value *call(SNode *snode,
                    llvm::Value *node_ptr,
                    const std::string &method,
                    const std::vector<llvm::Value *> &arguments) {
    auto prefix = get_runtime_snode_name(snode);
    auto s = emit_struct_meta(snode);
    auto s_ptr =
        builder->CreateBitCast(s, llvm::Type::getInt8PtrTy(*llvm_context));

    node_ptr = builder->CreateBitCast(node_ptr,
                                      llvm::Type::getInt8PtrTy(*llvm_context));

    std::vector<llvm::Value *> func_arguments{s_ptr, node_ptr};

    func_arguments.insert(func_arguments.end(), arguments.begin(),
                          arguments.end());

    return call(builder.get(), prefix + "_" + method, func_arguments);
  }

  void visit(GetRootStmt *stmt) override {
    stmt->value = builder->CreateBitCast(
        get_root(),
        PointerType::get(snode_attr[prog->snode_root.get()].llvm_type, 0));
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    auto shifted = builder->CreateAdd(stmt->input->value,
                                      tlctx->get_constant((int32)stmt->offset));
    int mask = (1u << (stmt->bit_end - stmt->bit_begin)) - 1;
    stmt->value =
        builder->CreateAnd(builder->CreateLShr(shifted, stmt->bit_begin),
                           tlctx->get_constant(mask));
  }

  void visit(LinearizeStmt *stmt) override {
    llvm::Value *val = tlctx->get_constant(0);
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = builder->CreateAdd(
          builder->CreateMul(val, tlctx->get_constant(stmt->strides[i])),
          stmt->inputs[i]->value);
    }
    stmt->value = val;
  }

  void visit(IntegerOffsetStmt *stmt) override {
    TI_NOT_IMPLEMENTED
    if (stmt->input->is<GetChStmt>() &&
        stmt->input->as<GetChStmt>()->output_snode->type == SNodeType::place) {
      auto input = stmt->input->as<GetChStmt>();
      auto dtn = input->output_snode->data_type_name();
      emit(R"({}* {}[1] {{({} *)((char *){}[0] + {})}};)", dtn,
           stmt->raw_name(), dtn, stmt->input->raw_name(), stmt->offset);
    } else {
      emit(R"(auto {} = {} + {};)", stmt->raw_name(), stmt->input->raw_name(),
           stmt->offset);
    }
  }

  void visit(SNodeLookupStmt *stmt) override {
    llvm::Value *parent = nullptr;
    parent = stmt->input_snode->value;
    TI_ASSERT(parent);
    auto snode = stmt->snode;
    if (snode->type == SNodeType::root) {
      stmt->value = builder->CreateGEP(parent, stmt->input_index->value);
    } else if (snode->type == SNodeType::dense ||
               snode->type == SNodeType::pointer ||
               snode->type == SNodeType::dynamic) {
      if (stmt->activate) {
        call(snode, stmt->input_snode->value, "activate",
             {stmt->input_index->value});
      }
      stmt->value = call(snode, stmt->input_snode->value, "lookup_element",
                         {stmt->input_index->value});
    } else {
      TI_INFO(snode_type_name(snode->type));
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(GetChStmt *stmt) override {
    auto ch = create_call(
        stmt->output_snode->get_ch_from_parent_func_name(),
        {builder->CreateBitCast(stmt->input_ptr->value,
                                PointerType::getInt8PtrTy(*llvm_context))});
    stmt->value = builder->CreateBitCast(
        ch, PointerType::get(snode_attr[stmt->output_snode].llvm_type, 0));
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);

    auto argload = stmt->base_ptrs[0]->as<ArgLoadStmt>();
    auto arg_id = argload->arg_id;
    int num_indices = stmt->indices.size();
    std::vector<llvm::Value *> sizes(num_indices);

    for (int i = 0; i < num_indices; i++) {
      auto raw_arg = builder->CreateCall(
          get_runtime_function("Context_get_extra_args"),
          {get_context(), tlctx->get_constant(arg_id), tlctx->get_constant(i)});
      sizes[i] = raw_arg;
    }

    auto dt = stmt->ret_type.data_type;
    auto base = builder->CreateBitCast(
        stmt->base_ptrs[0]->value,
        llvm::PointerType::get(tlctx->get_data_type(dt), 0));

    auto linear_index = tlctx->get_constant(0);
    for (int i = 0; i < num_indices; i++) {
      linear_index = builder->CreateMul(linear_index, sizes[i]);
      linear_index = builder->CreateAdd(linear_index, stmt->indices[i]->value);
    }

    stmt->value = builder->CreateGEP(base, linear_index);
  }

  BasicBlock *func_body_bb;

  virtual bool kernel_argument_by_val() const {
    return false;  // on CPU devices just pass in a pointer
  }

  std::string init_offloaded_task_function(OffloadedStmt *stmt,
                                           std::string suffix = "") {
    while_after_loop = nullptr;
    current_offloaded_stmt = stmt;

    task_function_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                {PointerType::get(context_ty, 0)}, false);

    auto task_kernel_name = fmt::format("{}_{}_{}{}", kernel_name, task_counter,
                                        stmt->task_name(), suffix);
    task_counter += 1;
    func = Function::Create(task_function_type, Function::ExternalLinkage,
                            task_kernel_name, module.get());

    current_task = std::make_unique<OffloadedTask>(this);
    current_task->begin(task_kernel_name);

    for (auto &arg : func->args()) {
      kernel_args.push_back(&arg);
    }
    kernel_args[0]->setName("context");

    if (kernel_argument_by_val())
      func->addParamAttr(0, llvm::Attribute::ByVal);

    // entry_block has all the allocas
    this->entry_block = BasicBlock::Create(*llvm_context, "entry", func);

    // The real function body
    func_body_bb = BasicBlock::Create(*llvm_context, "body", func);
    builder->SetInsertPoint(func_body_bb);
    return task_kernel_name;
  }

  void finalize_offloaded_task_function() {
    builder->CreateRetVoid();

    // entry_block should jump to the body after all allocas are inserted
    builder->SetInsertPoint(entry_block);
    builder->CreateBr(func_body_bb);

    if (prog->config.print_kernel_llvm_ir) {
      TI_INFO("Kernel Module IR");
      module->print(errs(), nullptr);
    }
    TI_ASSERT(!llvm::verifyFunction(*func, &errs()));
    // TI_INFO("Kernel function verified.");
  }

  class FunctionCreationGuard {
   public:
    CodeGenLLVM *mb;
    llvm::Function *old_func;
    llvm::Function *body;
    llvm::BasicBlock *old_entry, *allocas, *entry;
    llvm::IRBuilder<>::InsertPoint ip;

    FunctionCreationGuard(CodeGenLLVM *mb, std::vector<llvm::Type *> arguments)
        : mb(mb) {
      // Create the loop body function
      auto body_function_type = llvm::FunctionType::get(
          llvm::Type::getVoidTy(*mb->llvm_context), arguments, false);

      body = llvm::Function::Create(body_function_type,
                                    llvm::Function::InternalLinkage,
                                    "loop_body", mb->module.get());
      old_func = mb->func;
      // emit into loop body function
      mb->func = body;

      allocas = BasicBlock::Create(*mb->llvm_context, "allocs", body);
      old_entry = mb->entry_block;
      mb->entry_block = allocas;

      entry = BasicBlock::Create(*mb->llvm_context, "entry", mb->func);

      ip = mb->builder->saveIP();
      mb->builder->SetInsertPoint(entry);

      auto body_bb =
          BasicBlock::Create(*mb->llvm_context, "loop_body", mb->func);
      mb->builder->CreateBr(body_bb);
      mb->builder->SetInsertPoint(body_bb);
    }

    ~FunctionCreationGuard() {
      mb->builder->CreateRetVoid();
      mb->func = old_func;
      mb->builder->restoreIP(ip);

      {
        llvm::IRBuilderBase::InsertPointGuard gurad(*mb->builder);
        mb->builder->SetInsertPoint(allocas);
        mb->builder->CreateBr(entry);
        mb->entry_block = old_entry;
      }
    }
  };

  FunctionCreationGuard get_function_creation_gurad(
      std::vector<llvm::Type *> argument_types) {
    return FunctionCreationGuard(this, argument_types);
  }

  auto get_range_for_bounds(OffloadedStmt *stmt) {
    llvm::Value *begin, *end;
    if (stmt->const_begin) {
      begin = tlctx->get_constant(stmt->begin_value);
    } else {
      auto begin_stmt = Stmt::make<GlobalTemporaryStmt>(
          stmt->begin_offset, VectorType(1, DataType::i32));
      begin_stmt->accept(this);
      begin = builder->CreateLoad(begin_stmt->value);
    }
    if (stmt->const_end) {
      end = tlctx->get_constant(stmt->end_value);
    } else {
      auto end_stmt = Stmt::make<GlobalTemporaryStmt>(
          stmt->end_offset, VectorType(1, DataType::i32));
      end_stmt->accept(this);
      end = builder->CreateLoad(end_stmt->value);
    }
    return std::tuple(begin, end);
  }

  void create_offload_range_for(OffloadedStmt *stmt) {
    int step = 1;
    if (stmt->reversed) {
      step = -1;
    }

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

    auto [begin, end] = get_range_for_bounds(stmt);
    create_call("cpu_parallel_range_for",
                {get_arg(0), tlctx->get_constant(stmt->num_cpu_threads), begin,
                 end, tlctx->get_constant(step),
                 tlctx->get_constant(stmt->block_dim), body});
  }

  void create_offload_struct_for(OffloadedStmt *stmt, bool spmd = false) {
    llvm::Function *body;
    auto leaf_block = stmt->snode;
    {
      // Create the loop body function
      auto guard = get_function_creation_gurad({
          llvm::PointerType::get(get_runtime_type("Context"), 0),
          llvm::PointerType::get(get_runtime_type("Element"), 0),
          tlctx->get_data_type<int>(),
          tlctx->get_data_type<int>(),
      });

      body = guard.body;

      // per-leaf-block for loop
      auto loop_index =
          create_entry_block_alloca(Type::getInt32Ty(*llvm_context));

      llvm::Value *threadIdx = nullptr, *blockDim = nullptr;

      RuntimeObject element("Element", this, builder.get(), get_arg(1));
      auto lower_bound = get_arg(2);
      auto upper_bound = get_arg(3);
      // create_print("lower", DataType::i32, lower_bound);
      // create_print("upper", DataType::i32, upper_bound);

      if (spmd) {
        threadIdx = builder->CreateIntrinsic(
            Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
        blockDim = builder->CreateIntrinsic(
            Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});
        builder->CreateStore(builder->CreateAdd(threadIdx, lower_bound),
                             loop_index);
      } else {
        builder->CreateStore(lower_bound, loop_index);
      }

      // test bb
      auto test_bb = BasicBlock::Create(*llvm_context, "test", func);
      auto body_bb = BasicBlock::Create(*llvm_context, "loop_body", func);
      auto after_loop = BasicBlock::Create(*llvm_context, "after_loop", func);

      builder->CreateBr(test_bb);
      {
        builder->SetInsertPoint(test_bb);
        auto cond =
            builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                builder->CreateLoad(loop_index), upper_bound);
        builder->CreateCondBr(cond, body_bb, after_loop);
      }

      builder->SetInsertPoint(body_bb);

      // initialize the coordinates
      auto refine =
          get_runtime_function(leaf_block->refine_coordinates_func_name());
      auto new_coordinates = create_entry_block_alloca(physical_coordinate_ty);
      create_call(refine, {element.get_ptr("pcoord"), new_coordinates,
                           builder->CreateLoad(loop_index)});

      current_coordinates = new_coordinates;

      // Additional compare if non-POT exists
      auto nonpot_cond = tlctx->get_constant(true);
      auto snode = stmt->snode;

      auto coord_object = RuntimeObject("PhysicalCoordinates", this,
                                        builder.get(), new_coordinates);
      for (int i = 0; i < snode->num_active_indices; i++) {
        auto j = snode->physical_index_position[i];
        if (!bit::is_power_of_two(snode->extractors[j].num_elements)) {
          auto coord = coord_object.get("val", tlctx->get_constant(j));
          nonpot_cond = builder->CreateAnd(
              nonpot_cond,
              builder->CreateICmp(
                  llvm::CmpInst::ICMP_SLT, coord,
                  tlctx->get_constant(snode->extractors[j].num_elements)));
        }
      }

      auto body_bb_tail =
          BasicBlock::Create(*llvm_context, "loop_body_tail", func);
      {
        auto bounded_body_bb =
            BasicBlock::Create(*llvm_context, "bound_guarded_loop_body", func);
        builder->CreateCondBr(nonpot_cond, bounded_body_bb, body_bb_tail);
        builder->SetInsertPoint(bounded_body_bb);
        // The real loop body
        stmt->body->accept(this);
        builder->CreateBr(body_bb_tail);
      }

      // body cfg

      builder->SetInsertPoint(body_bb_tail);

      if (spmd) {
        create_increment(loop_index, blockDim);
      } else {
        create_increment(loop_index, tlctx->get_constant(1));
      }
      builder->CreateBr(test_bb);

      builder->SetInsertPoint(after_loop);
      builder->CreateRetVoid();
    }

    if (stmt->block_dim == 0) {
      stmt->block_dim = std::min(leaf_block->max_num_elements(), 256);
    }
    int num_splits = leaf_block->max_num_elements() / stmt->block_dim;
    // traverse leaf node
    create_call("for_each_block",
                {get_context(), tlctx->get_constant(leaf_block->id),
                 tlctx->get_constant(leaf_block->max_num_elements()),
                 tlctx->get_constant(num_splits), body,
                 tlctx->get_constant(stmt->num_cpu_threads)});
  }

  void visit(LoopIndexStmt *stmt) override {
    if (stmt->is_struct_for) {
      stmt->value = builder->CreateLoad(builder->CreateGEP(
          current_coordinates, {tlctx->get_constant(0), tlctx->get_constant(0),
                                tlctx->get_constant(stmt->index)}));
    } else {
      stmt->value = builder->CreateLoad(
          current_offloaded_stmt->loop_vars_llvm[stmt->index]);
    }
  }

  void visit(GlobalTemporaryStmt *stmt) override {
    auto runtime = get_runtime();
    auto buffer = call("get_temporary_pointer", runtime,
                       tlctx->get_constant((int64)stmt->offset));

    TI_ASSERT(stmt->width() == 1);
    auto ptr_type = llvm::PointerType::get(
        tlctx->get_data_type(stmt->ret_type.data_type), 0);
    stmt->value = builder->CreatePointerCast(buffer, ptr_type);
  }

  void visit(InternalFuncStmt *stmt) override {
    create_call(stmt->func_name, {get_context()});
  }

  ~CodeGenLLVM() {
  }
};

TLANG_NAMESPACE_END
