#include "codegen_llvm.h"

#include "taichi/struct/struct_llvm.h"

TLANG_NAMESPACE_BEGIN

// TODO: sort function definitions to match declaration order in header

// OffloadedTask

OffloadedTask::OffloadedTask(CodeGenLLVM *codegen) : codegen(codegen) {
  func = nullptr;
}

void OffloadedTask::begin(const std::string &name) {
  this->name = name;
}

void OffloadedTask::end() {
  codegen->offloaded_tasks.push_back(*this);
}

void OffloadedTask::operator()(Context *context) {
  TI_ASSERT(func);
  func(context);
}

void OffloadedTask::compile() {
  TI_ASSERT(!func);
  auto kernel_symbol = codegen->tlctx->lookup_function_pointer(name);
  TI_ASSERT_INFO(kernel_symbol, "Function not found");

  func = (task_fp_type)kernel_symbol;
}

// TODO(k-ye): Hide FunctionCreationGuard inside cpp file
FunctionCreationGuard::FunctionCreationGuard(
    CodeGenLLVM *mb,
    std::vector<llvm::Type *> arguments)
    : mb(mb) {
  // Create the loop body function
  auto body_function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*mb->llvm_context), arguments, false);

  body = llvm::Function::Create(body_function_type,
                                llvm::Function::InternalLinkage, "loop_body",
                                mb->module.get());
  old_func = mb->func;
  // emit into loop body function
  mb->func = body;

  allocas = BasicBlock::Create(*mb->llvm_context, "allocs", body);
  old_entry = mb->entry_block;
  mb->entry_block = allocas;

  entry = BasicBlock::Create(*mb->llvm_context, "entry", mb->func);

  ip = mb->builder->saveIP();
  mb->builder->SetInsertPoint(entry);

  auto body_bb = BasicBlock::Create(*mb->llvm_context, "loop_body", mb->func);
  mb->builder->CreateBr(body_bb);
  mb->builder->SetInsertPoint(body_bb);
}

FunctionCreationGuard::~FunctionCreationGuard() {
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

namespace {

class CodeGenStmtGuard {
 public:
  using Getter = std::function<llvm::BasicBlock *(void)>;
  using Setter = std::function<void(llvm::BasicBlock *)>;

  explicit CodeGenStmtGuard(Getter getter, Setter setter)
      : saved_stmt_(getter()), setter_(std::move(setter)) {
  }

  ~CodeGenStmtGuard() {
    setter_(saved_stmt_);
  }

  CodeGenStmtGuard(CodeGenStmtGuard &&) = default;
  CodeGenStmtGuard &operator=(CodeGenStmtGuard &&) = default;

 private:
  llvm::BasicBlock *saved_stmt_;
  Setter setter_;
};

CodeGenStmtGuard make_loop_reentry_guard(CodeGenLLVM *cg) {
  return CodeGenStmtGuard([cg]() { return cg->current_loop_reentry; },
                          [cg](llvm::BasicBlock *saved_stmt) {
                            cg->current_loop_reentry = saved_stmt;
                          });
}

CodeGenStmtGuard make_while_after_loop_guard(CodeGenLLVM *cg) {
  return CodeGenStmtGuard([cg]() { return cg->current_while_after_loop; },
                          [cg](llvm::BasicBlock *saved_stmt) {
                            cg->current_while_after_loop = saved_stmt;
                          });
}

}  // namespace

// CodeGenLLVM

uint64 CodeGenLLVM::task_counter = 0;

void CodeGenLLVM::visit(Block *stmt_list) {
  for (auto &stmt : stmt_list->statements) {
    stmt->accept(this);
  }
}

void CodeGenLLVM::visit(AllocaStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);
  llvm_val[stmt] = create_entry_block_alloca(stmt->ret_type.data_type);
  // initialize as zero
  builder->CreateStore(tlctx->get_constant(stmt->ret_type.data_type, 0),
                       llvm_val[stmt]);
}

void CodeGenLLVM::visit(RandStmt *stmt) {
  llvm_val[stmt] = create_call(
      fmt::format("rand_{}", data_type_short_name(stmt->ret_type.data_type)));
}

void CodeGenLLVM::emit_extra_unary(UnaryOpStmt *stmt) {
  auto input = llvm_val[stmt->operand];
  auto input_taichi_type = stmt->operand->ret_type.data_type;
  auto op = stmt->op_type;
  auto input_type = input->getType();

#define UNARY_STD(x)                                                   \
  else if (op == UnaryOpType::x) {                                     \
    if (input_taichi_type == DataType::f32) {                          \
      llvm_val[stmt] =                                                 \
          builder->CreateCall(get_runtime_function(#x "_f32"), input); \
    } else if (input_taichi_type == DataType::f64) {                   \
      llvm_val[stmt] =                                                 \
          builder->CreateCall(get_runtime_function(#x "_f64"), input); \
    } else if (input_taichi_type == DataType::i32) {                   \
      llvm_val[stmt] =                                                 \
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
    llvm_val[stmt] =
        builder->CreateIntrinsic(llvm::Intrinsic::sqrt, {input_type}, {input});
  }
  else {
    TI_P(unary_op_type_name(op));
    TI_NOT_IMPLEMENTED
  }
#undef UNARY_STD
}

std::unique_ptr<RuntimeObject> CodeGenLLVM::emit_struct_meta_object(
    SNode *snode) {
  std::unique_ptr<RuntimeObject> meta;
  if (snode->type == SNodeType::dense) {
    meta = std::make_unique<RuntimeObject>("DenseMeta", this, builder.get());
    emit_struct_meta_base("Dense", meta->ptr, snode);
    meta->call("set_morton_dim", tlctx->get_constant((int)snode->_morton));
  } else if (snode->type == SNodeType::pointer) {
    meta = std::make_unique<RuntimeObject>("PointerMeta", this, builder.get());
    emit_struct_meta_base("Pointer", meta->ptr, snode);
  } else if (snode->type == SNodeType::root) {
    meta = std::make_unique<RuntimeObject>("RootMeta", this, builder.get());
    emit_struct_meta_base("Root", meta->ptr, snode);
  } else if (snode->type == SNodeType::dynamic) {
    meta = std::make_unique<RuntimeObject>("DynamicMeta", this, builder.get());
    emit_struct_meta_base("Dynamic", meta->ptr, snode);
    meta->call("set_chunk_size", tlctx->get_constant(snode->chunk_size));
  } else if (snode->type == SNodeType::bitmasked) {
    meta =
        std::make_unique<RuntimeObject>("BitmaskedMeta", this, builder.get());
    emit_struct_meta_base("Bitmasked", meta->ptr, snode);
  } else {
    TI_P(snode_type_name(snode->type));
    TI_NOT_IMPLEMENTED;
  }
  return meta;
}

void CodeGenLLVM::emit_struct_meta_base(const std::string &name,
                                        llvm::Value *node_meta,
                                        SNode *snode) {
  RuntimeObject common("StructMeta", this, builder.get(), node_meta);
  std::size_t element_size;
  if (snode->type == SNodeType::dense) {
    auto body_type =
        StructCompilerLLVM::get_llvm_body_type(module.get(), snode);
    auto element_ty = body_type->getArrayElementType();
    element_size = tlctx->get_type_size(element_ty);
  } else if (snode->type == SNodeType::pointer) {
    auto element_ty = StructCompilerLLVM::get_llvm_node_type(
        module.get(), snode->ch[0].get());
    element_size = tlctx->get_type_size(element_ty);
  } else {
    auto element_ty =
        StructCompilerLLVM::get_llvm_element_type(module.get(), snode);
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

  for (auto const &f : functions)
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

CodeGenLLVM::CodeGenLLVM(Kernel *kernel, IRNode *ir)
    // TODO: simplify LLVMModuleBuilder ctor input
    : LLVMModuleBuilder(
          kernel->program.get_llvm_context(kernel->arch)->clone_struct_module(),
          kernel->program.get_llvm_context(kernel->arch)),
      kernel(kernel),
      ir(ir),
      prog(&kernel->program) {
  if (ir == nullptr)
    this->ir = kernel->ir;
  initialize_context();
  current_offloaded_stmt = nullptr;

  context_ty = get_runtime_type("Context");
  physical_coordinate_ty = get_runtime_type("PhysicalCoordinates");

  kernel_name = kernel->name + "_kernel";
}

void CodeGenLLVM::visit(UnaryOpStmt *stmt) {
  auto input = llvm_val[stmt->operand];
  auto input_type = input->getType();
  auto op = stmt->op_type;

#define UNARY_INTRINSIC(x)                                                   \
  else if (op == UnaryOpType::x) {                                           \
    llvm_val[stmt] =                                                         \
        builder->CreateIntrinsic(llvm::Intrinsic::x, {input_type}, {input}); \
  }
  if (stmt->op_type == UnaryOpType::cast_value) {
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
      llvm_val[stmt] =
          builder->CreateCast(cast_op, llvm_val[stmt->operand],
                              tlctx->get_data_type(stmt->cast_type));
    } else if (is_real(from) && is_real(to)) {
      if (data_type_size(from) < data_type_size(to)) {
        llvm_val[stmt] = builder->CreateFPExt(
            llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
      } else {
        llvm_val[stmt] = builder->CreateFPTrunc(
            llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
      }
    } else if (!is_real(from) && !is_real(to)) {
      if (data_type_size(from) < data_type_size(to)) {
        llvm_val[stmt] = builder->CreateSExt(
            llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
      } else {
        llvm_val[stmt] = builder->CreateTrunc(
            llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
      }
    }
  } else if (stmt->op_type == UnaryOpType::cast_bits) {
    TI_ASSERT(data_type_size(stmt->ret_type.data_type) ==
              data_type_size(stmt->cast_type));
    llvm_val[stmt] = builder->CreateBitCast(
        llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
  } else if (op == UnaryOpType::rsqrt) {
    llvm::Function *sqrt_fn = Intrinsic::getDeclaration(
        module.get(), Intrinsic::sqrt, input->getType());
    auto intermediate = builder->CreateCall(sqrt_fn, input, "sqrt");
    llvm_val[stmt] = builder->CreateFDiv(
        tlctx->get_constant(stmt->ret_type.data_type, 1.0), intermediate);
  } else if (op == UnaryOpType::bit_not) {
    llvm_val[stmt] = builder->CreateNot(input);
  } else if (op == UnaryOpType::neg) {
    if (is_real(stmt->operand->ret_type.data_type)) {
      llvm_val[stmt] = builder->CreateFNeg(input, "neg");
    } else {
      llvm_val[stmt] = builder->CreateNeg(input, "neg");
    }
  }
  UNARY_INTRINSIC(floor)
  UNARY_INTRINSIC(ceil)
  else emit_extra_unary(stmt);
#undef UNARY_INTRINSIC
}

void CodeGenLLVM::visit(BinaryOpStmt *stmt) {
  auto op = stmt->op_type;
  auto ret_type = stmt->ret_type.data_type;
  if (op == BinaryOpType::add) {
    if (is_real(stmt->ret_type.data_type)) {
      llvm_val[stmt] =
          builder->CreateFAdd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] =
          builder->CreateAdd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::sub) {
    if (is_real(stmt->ret_type.data_type)) {
      llvm_val[stmt] =
          builder->CreateFSub(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] =
          builder->CreateSub(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::mul) {
    if (is_real(stmt->ret_type.data_type)) {
      llvm_val[stmt] =
          builder->CreateFMul(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] =
          builder->CreateMul(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::floordiv) {
    if (is_integral(ret_type))
      llvm_val[stmt] = create_call(
          fmt::format("floordiv_{}", data_type_short_name(ret_type)),
          {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
    else {
      auto div = builder->CreateFDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      llvm_val[stmt] = builder->CreateIntrinsic(
          llvm::Intrinsic::floor, {tlctx->get_data_type(ret_type)}, {div});
    }
  } else if (op == BinaryOpType::div) {
    if (is_real(stmt->ret_type.data_type)) {
      llvm_val[stmt] =
          builder->CreateFDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] =
          builder->CreateSDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::mod) {
    llvm_val[stmt] =
        builder->CreateSRem(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_and) {
    llvm_val[stmt] =
        builder->CreateAnd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_or) {
    llvm_val[stmt] =
        builder->CreateOr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_xor) {
    llvm_val[stmt] =
        builder->CreateXor(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::max) {
    if (is_real(ret_type)) {
      llvm_val[stmt] =
          builder->CreateMaxNum(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else if (ret_type == DataType::i32) {
      llvm_val[stmt] =
          create_call("max_i32", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
    } else {
      TI_P(data_type_name(ret_type));
      TI_NOT_IMPLEMENTED
    }
  } else if (op == BinaryOpType::atan2) {
    if (arch_is_cpu(current_arch())) {
      if (ret_type == DataType::f32) {
        llvm_val[stmt] = create_call(
            "atan2_f32", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::f64) {
        llvm_val[stmt] = create_call(
            "atan2_f64", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else if (current_arch() == Arch::cuda) {
      if (ret_type == DataType::f32) {
        llvm_val[stmt] = create_call(
            "__nv_atan2f", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::f64) {
        llvm_val[stmt] = create_call(
            "__nv_atan2", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
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
        llvm_val[stmt] =
            create_call("pow_f32", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::f64) {
        llvm_val[stmt] =
            create_call("pow_f64", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::i32) {
        llvm_val[stmt] =
            create_call("pow_i32", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::i64) {
        llvm_val[stmt] =
            create_call("pow_i64", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else if (current_arch() == Arch::cuda) {
      if (ret_type == DataType::f32) {
        llvm_val[stmt] = create_call(
            "__nv_powf", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::f64) {
        llvm_val[stmt] =
            create_call("__nv_pow", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::i32) {
        llvm_val[stmt] =
            create_call("pow_i32", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else if (ret_type == DataType::i64) {
        llvm_val[stmt] =
            create_call("pow_i64", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    } else {
      TI_NOT_IMPLEMENTED
    }
  } else if (op == BinaryOpType::min) {
    if (is_real(ret_type)) {
      llvm_val[stmt] =
          builder->CreateMinNum(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else if (ret_type == DataType::i32) {
      llvm_val[stmt] =
          create_call("min_i32", {llvm_val[stmt->lhs], llvm_val[stmt->rhs]});
    } else {
      TI_P(data_type_name(ret_type));
      TI_NOT_IMPLEMENTED
    }
  } else if (is_comparison(op)) {
    llvm::Value *cmp = nullptr;
    auto input_type = stmt->lhs->ret_type.data_type;
    if (op == BinaryOpType::cmp_eq) {
      if (is_real(input_type)) {
        cmp = builder->CreateFCmpOEQ(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        cmp = builder->CreateICmpEQ(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      }
    } else if (op == BinaryOpType::cmp_le) {
      if (is_real(input_type)) {
        cmp = builder->CreateFCmpOLE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type)) {
          cmp =
              builder->CreateICmpSLE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpULE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_ge) {
      if (is_real(input_type)) {
        cmp = builder->CreateFCmpOGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type)) {
          cmp =
              builder->CreateICmpSGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpUGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_lt) {
      if (is_real(input_type)) {
        cmp = builder->CreateFCmpOLT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type)) {
          cmp =
              builder->CreateICmpSLT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpULT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_gt) {
      if (is_real(input_type)) {
        cmp = builder->CreateFCmpOGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type)) {
          cmp =
              builder->CreateICmpSGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpUGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_ne) {
      if (is_real(input_type)) {
        cmp = builder->CreateFCmpONE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        cmp = builder->CreateICmpNE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      }
    } else {
      TI_NOT_IMPLEMENTED
    }
    llvm_val[stmt] = builder->CreateSExt(cmp, llvm_type(DataType::i32));
  } else {
    TI_P(binary_op_type_name(op));
    TI_NOT_IMPLEMENTED
  }
}

llvm::Type *CodeGenLLVM::llvm_type(DataType dt) {
  if (dt == DataType::i32) {
    return llvm::Type::getInt32Ty(*llvm_context);
  } else if (dt == DataType::u1) {
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

void CodeGenLLVM::visit(TernaryOpStmt *stmt) {
  TI_ASSERT(stmt->op_type == TernaryOpType::select);
  llvm_val[stmt] = builder->CreateSelect(
      builder->CreateTrunc(llvm_val[stmt->op1], llvm_type(DataType::u1)),
      llvm_val[stmt->op2], llvm_val[stmt->op3]);
}

void CodeGenLLVM::visit(IfStmt *if_stmt) {
  // TODO: take care of vectorized cases
  BasicBlock *true_block =
      BasicBlock::Create(*llvm_context, "true_block", func);
  BasicBlock *false_block =
      BasicBlock::Create(*llvm_context, "false_block", func);
  BasicBlock *after_if = BasicBlock::Create(*llvm_context, "after_if", func);
  builder->CreateCondBr(
      builder->CreateICmpNE(llvm_val[if_stmt->cond], tlctx->get_constant(0)),
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

llvm::Value *CodeGenLLVM::create_print(std::string tag,
                                       DataType dt,
                                       llvm::Value *value) {
  TI_ASSERT(arch_use_host_memory(kernel->arch));
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
  auto runtime_printf = call("LLVMRuntime_get_host_printf", get_runtime());
  args.push_back(builder->CreateGlobalStringPtr(
      ("[llvm codegen debug] " + tag + " = " + format + "\n").c_str(),
      "format_string"));
  args.push_back(value);
  return builder->CreateCall(runtime_printf, args);
}

void CodeGenLLVM::visit(PrintStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);
  std::vector<Value *> args;
  std::string format;
  auto value = llvm_val[stmt->stmt];
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

  llvm_val[stmt] = builder->CreateCall(runtime_printf, args);
}

void CodeGenLLVM::visit(ConstStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);
  auto val = stmt->val[0];
  if (val.dt == DataType::f32) {
    llvm_val[stmt] =
        llvm::ConstantFP::get(*llvm_context, llvm::APFloat(val.val_float32()));
  } else if (val.dt == DataType::f64) {
    llvm_val[stmt] =
        llvm::ConstantFP::get(*llvm_context, llvm::APFloat(val.val_float64()));
  } else if (val.dt == DataType::i32) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(32, val.val_int32(), true));
  } else if (val.dt == DataType::i64) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(64, val.val_int64(), true));
  } else {
    TI_P(data_type_name(val.dt));
    TI_NOT_IMPLEMENTED;
  }
}

void CodeGenLLVM::visit(WhileControlStmt *stmt) {
  BasicBlock *after_break =
      BasicBlock::Create(*llvm_context, "after_break", func);
  TI_ASSERT(current_while_after_loop);
  auto cond =
      builder->CreateICmpEQ(llvm_val[stmt->cond], tlctx->get_constant(0));
  builder->CreateCondBr(cond, current_while_after_loop, after_break);
  builder->SetInsertPoint(after_break);
}

void CodeGenLLVM::visit(ContinueStmt *stmt) {
  if (stmt->as_return()) {
    builder->CreateRetVoid();
  } else {
    TI_ASSERT(current_loop_reentry != nullptr);
    builder->CreateBr(current_loop_reentry);
  }
  // Stmts after continue are useless, so we switch the insertion point to
  // /dev/null. In LLVM IR, the "after_continue" label shows "No predecessors!".
  BasicBlock *after_continue =
      BasicBlock::Create(*llvm_context, "after_continue", func);
  builder->SetInsertPoint(after_continue);
}

void CodeGenLLVM::visit(WhileStmt *stmt) {
  BasicBlock *body = BasicBlock::Create(*llvm_context, "while_loop_body", func);
  builder->CreateBr(body);
  builder->SetInsertPoint(body);
  auto lrg = make_loop_reentry_guard(this);
  current_loop_reentry = body;

  BasicBlock *after_loop =
      BasicBlock::Create(*llvm_context, "after_while", func);
  auto walg = make_while_after_loop_guard(this);
  current_while_after_loop = after_loop;

  stmt->body->accept(this);

  builder->CreateBr(body);  // jump to head

  builder->SetInsertPoint(after_loop);
}

llvm::Value *CodeGenLLVM::cast_pointer(llvm::Value *val,
                                       std::string dest_ty_name,
                                       int addr_space) {
  return builder->CreateBitCast(
      val, llvm::PointerType::get(get_runtime_type(dest_ty_name), addr_space));
}

void CodeGenLLVM::emit_clear_list(OffloadedStmt *listgen) {
  auto snode_child = listgen->snode;
  auto snode_parent = listgen->snode->parent;
  auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
  auto meta_parent = cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
  call("clear_list", get_runtime(), meta_parent, meta_child);
}

void CodeGenLLVM::emit_list_gen(OffloadedStmt *listgen) {
  auto snode_child = listgen->snode;
  auto snode_parent = listgen->snode->parent;
  auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
  auto meta_parent = cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
  call("element_listgen", get_runtime(), meta_parent, meta_child);
}

void CodeGenLLVM::emit_gc(OffloadedStmt *stmt) {
  auto snode = stmt->snode->id;
  call("node_gc", get_runtime(), tlctx->get_constant(snode));
}

llvm::Value *CodeGenLLVM::create_call(llvm::Value *func,
                                      std::vector<llvm::Value *> args) {
  check_func_call_signature(func, args);
  return builder->CreateCall(func, args);
}
llvm::Value *CodeGenLLVM::create_call(std::string func_name,
                                      std::vector<llvm::Value *> args) {
  auto func = get_runtime_function(func_name);
  return create_call(func, args);
}

void CodeGenLLVM::create_increment(llvm::Value *ptr, llvm::Value *value) {
  builder->CreateStore(builder->CreateAdd(builder->CreateLoad(ptr), value),
                       ptr);
}

void CodeGenLLVM::create_naive_range_for(RangeForStmt *for_stmt) {
  BasicBlock *body = BasicBlock::Create(*llvm_context, "for_loop_body", func);
  BasicBlock *loop_inc =
      BasicBlock::Create(*llvm_context, "for_loop_inc", func);
  BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "after_for", func);
  BasicBlock *loop_test =
      BasicBlock::Create(*llvm_context, "for_loop_test", func);

  auto loop_var = create_entry_block_alloca(DataType::i32);
  loop_vars_llvm[for_stmt].push_back(loop_var);

  if (!for_stmt->reversed) {
    builder->CreateStore(llvm_val[for_stmt->begin], loop_var);
  } else {
    builder->CreateStore(
        builder->CreateSub(llvm_val[for_stmt->end], tlctx->get_constant(1)),
        loop_var);
  }
  builder->CreateBr(loop_test);

  {
    // test block
    builder->SetInsertPoint(loop_test);
    llvm::Value *cond;
    if (!for_stmt->reversed) {
      cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT,
                                 builder->CreateLoad(loop_var),
                                 llvm_val[for_stmt->end]);
    } else {
      cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE,
                                 builder->CreateLoad(loop_var),
                                 llvm_val[for_stmt->begin]);
    }
    builder->CreateCondBr(cond, body, after_loop);
  }

  {
    {
      auto lrg = make_loop_reentry_guard(this);
      // The continue stmt should jump to the loop-increment block!
      current_loop_reentry = loop_inc;
      // body cfg
      builder->SetInsertPoint(body);

      for_stmt->body->accept(this);
    }

    builder->CreateBr(loop_inc);
    builder->SetInsertPoint(loop_inc);

    if (!for_stmt->reversed) {
      create_increment(loop_var, tlctx->get_constant(1));
    } else {
      create_increment(loop_var, tlctx->get_constant(-1));
    }
    builder->CreateBr(loop_test);
  }

  // next cfg
  builder->SetInsertPoint(after_loop);
}

void CodeGenLLVM::visit(RangeForStmt *for_stmt) {
  create_naive_range_for(for_stmt);
}

void CodeGenLLVM::visit(ArgLoadStmt *stmt) {
  auto raw_arg = call(builder.get(), "Context_get_args", get_context(),
                      tlctx->get_constant(stmt->arg_id));

  llvm::Type *dest_ty = nullptr;
  if (stmt->is_ptr) {
    dest_ty = PointerType::get(tlctx->get_data_type(DataType::i32), 0);
    llvm_val[stmt] = builder->CreateIntToPtr(raw_arg, dest_ty);
  } else {
    dest_ty = tlctx->get_data_type(stmt->ret_type.data_type);
    auto dest_bits = dest_ty->getPrimitiveSizeInBits();
    auto truncated = builder->CreateTrunc(
        raw_arg, Type::getIntNTy(*llvm_context, dest_bits));
    llvm_val[stmt] = builder->CreateBitCast(truncated, dest_ty);
  }
}

void CodeGenLLVM::visit(KernelReturnStmt *stmt) {
  if (stmt->is_ptr) {
    TI_NOT_IMPLEMENTED
  } else {
    auto intermediate_bits =
        tlctx->get_data_type(stmt->value->ret_type.data_type)
            ->getPrimitiveSizeInBits();
    llvm::Type *intermediate_type =
        llvm::Type::getIntNTy(*llvm_context, intermediate_bits);
    llvm::Type *dest_ty = tlctx->get_data_type<int64>();
    auto extended = builder->CreateZExt(
        builder->CreateBitCast(llvm_val[stmt->value], intermediate_type),
        dest_ty);
    builder->CreateCall(get_runtime_function("LLVMRuntime_store_result"),
                        {get_runtime(), extended});
  }
}

void CodeGenLLVM::visit(LocalLoadStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);
  llvm_val[stmt] = builder->CreateLoad(llvm_val[stmt->ptr[0].var]);
}

void CodeGenLLVM::visit(LocalStoreStmt *stmt) {
  auto mask = stmt->parent->mask();
  if (mask && stmt->width() != 1) {
    TI_NOT_IMPLEMENTED
  } else {
    builder->CreateStore(llvm_val[stmt->data], llvm_val[stmt->ptr]);
  }
}

void CodeGenLLVM::visit(AssertStmt *stmt) {
  if (stmt->args.empty()) {
    llvm_val[stmt] = call("taichi_assert", get_context(), llvm_val[stmt->cond],
                          builder->CreateGlobalStringPtr(stmt->text));
  } else {
    std::vector<Value *> args;
    args.emplace_back(get_runtime());
    args.emplace_back(llvm_val[stmt->cond]);
    args.emplace_back(builder->CreateGlobalStringPtr(stmt->text));
    for (auto arg : stmt->args) {
      TI_ASSERT(llvm_val[arg]);
      args.emplace_back(llvm_val[arg]);
    }
    llvm_val[stmt] = create_call("taichi_assert_format", args);
  }
}

void CodeGenLLVM::visit(SNodeOpStmt *stmt) {
  auto snode = stmt->snode;
  if (stmt->op_type == SNodeOpType::append) {
    TI_ASSERT(snode->type == SNodeType::dynamic);
    TI_ASSERT(stmt->ret_type.data_type == DataType::i32);
    llvm_val[stmt] =
        call(snode, llvm_val[stmt->ptr], "append", {llvm_val[stmt->val]});
  } else if (stmt->op_type == SNodeOpType::length) {
    TI_ASSERT(snode->type == SNodeType::dynamic);
    llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "get_num_elements", {});
  } else if (stmt->op_type == SNodeOpType::is_active) {
    llvm_val[stmt] =
        call(snode, llvm_val[stmt->ptr], "is_active", {llvm_val[stmt->val]});
  } else if (stmt->op_type == SNodeOpType::deactivate) {
    if (snode->type == SNodeType::pointer || snode->type == SNodeType::hash ||
        snode->type == SNodeType::bitmasked) {
      llvm_val[stmt] =
          call(snode, llvm_val[stmt->ptr], "deactivate", {llvm_val[stmt->val]});
    } else if (snode->type == SNodeType::dynamic) {
      llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "deactivate", {});
    }
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void CodeGenLLVM::visit(AtomicOpStmt *stmt) {
  // auto mask = stmt->parent->mask();
  // TODO: deal with mask when vectorized
  TI_ASSERT(stmt->width() == 1);
  for (int l = 0; l < stmt->width(); l++) {
    llvm::Value *old_value;
    if (stmt->op_type == AtomicOpType::add) {
      if (is_integral(stmt->val->ret_type.data_type)) {
        old_value = builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::BinOp::Add, llvm_val[stmt->dest],
            llvm_val[stmt->val], llvm::AtomicOrdering::SequentiallyConsistent);
      } else if (stmt->val->ret_type.data_type == DataType::f32) {
        old_value =
            builder->CreateCall(get_runtime_function("atomic_add_f32"),
                                {llvm_val[stmt->dest], llvm_val[stmt->val]});
      } else if (stmt->val->ret_type.data_type == DataType::f64) {
        old_value =
            builder->CreateCall(get_runtime_function("atomic_add_f64"),
                                {llvm_val[stmt->dest], llvm_val[stmt->val]});
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == AtomicOpType::min) {
      if (is_integral(stmt->val->ret_type.data_type)) {
        old_value = builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::BinOp::Min, llvm_val[stmt->dest],
            llvm_val[stmt->val], llvm::AtomicOrdering::SequentiallyConsistent);
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
            llvm_val[stmt->val], llvm::AtomicOrdering::SequentiallyConsistent);
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
            llvm_val[stmt->val], llvm::AtomicOrdering::SequentiallyConsistent);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == AtomicOpType::bit_or) {
      if (is_integral(stmt->val->ret_type.data_type)) {
        old_value = builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::BinOp::Or, llvm_val[stmt->dest],
            llvm_val[stmt->val], llvm::AtomicOrdering::SequentiallyConsistent);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (stmt->op_type == AtomicOpType::bit_xor) {
      if (is_integral(stmt->val->ret_type.data_type)) {
        old_value = builder->CreateAtomicRMW(
            llvm::AtomicRMWInst::BinOp::Xor, llvm_val[stmt->dest],
            llvm_val[stmt->val], llvm::AtomicOrdering::SequentiallyConsistent);
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else {
      TI_NOT_IMPLEMENTED
    }
    llvm_val[stmt] = old_value;
  }
}

void CodeGenLLVM::visit(GlobalPtrStmt *stmt) {
  TI_ERROR("Global Ptrs should have been lowered.");
}

void CodeGenLLVM::visit(GlobalStoreStmt *stmt) {
  TI_ASSERT(!stmt->parent->mask() || stmt->width() == 1);
  TI_ASSERT(llvm_val[stmt->data]);
  TI_ASSERT(llvm_val[stmt->ptr]);
  builder->CreateStore(llvm_val[stmt->data], llvm_val[stmt->ptr]);
}

void CodeGenLLVM::visit(GlobalLoadStmt *stmt) {
  int width = stmt->width();
  TI_ASSERT(width == 1);
  llvm_val[stmt] = builder->CreateLoad(
      tlctx->get_data_type(stmt->ret_type.data_type), llvm_val[stmt->ptr]);
}

void CodeGenLLVM::visit(ElementShuffleStmt *stmt){
    TI_NOT_IMPLEMENTED
    /*
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
    */
}

std::string CodeGenLLVM::get_runtime_snode_name(SNode *snode) {
  if (snode->type == SNodeType::root) {
    return "Root";
  } else if (snode->type == SNodeType::dense) {
    return "Dense";
  } else if (snode->type == SNodeType::dynamic) {
    return "Dynamic";
  } else if (snode->type == SNodeType::pointer) {
    return "Pointer";
  } else if (snode->type == SNodeType::hash) {
    return "Hash";
  } else if (snode->type == SNodeType::bitmasked) {
    return "Bitmasked";
  } else {
    TI_P(snode_type_name(snode->type));
    TI_NOT_IMPLEMENTED
  }
}

llvm::Value *CodeGenLLVM::call(SNode *snode,
                               llvm::Value *node_ptr,
                               const std::string &method,
                               const std::vector<llvm::Value *> &arguments) {
  auto prefix = get_runtime_snode_name(snode);
  auto s = emit_struct_meta(snode);
  auto s_ptr =
      builder->CreateBitCast(s, llvm::Type::getInt8PtrTy(*llvm_context));

  node_ptr =
      builder->CreateBitCast(node_ptr, llvm::Type::getInt8PtrTy(*llvm_context));

  std::vector<llvm::Value *> func_arguments{s_ptr, node_ptr};

  func_arguments.insert(func_arguments.end(), arguments.begin(),
                        arguments.end());

  return call(builder.get(), prefix + "_" + method, func_arguments);
}

void CodeGenLLVM::visit(GetRootStmt *stmt) {
  llvm_val[stmt] = builder->CreateBitCast(
      get_root(), PointerType::get(StructCompilerLLVM::get_llvm_node_type(
                                       module.get(), prog->snode_root.get()),
                                   0));
}

void CodeGenLLVM::visit(OffsetAndExtractBitsStmt *stmt) {
  auto shifted = builder->CreateAdd(llvm_val[stmt->input],
                                    tlctx->get_constant((int32)stmt->offset));
  int mask = (1u << (stmt->bit_end - stmt->bit_begin)) - 1;
  llvm_val[stmt] = builder->CreateAnd(
      builder->CreateLShr(shifted, stmt->bit_begin), tlctx->get_constant(mask));
}

void CodeGenLLVM::visit(LinearizeStmt *stmt) {
  llvm::Value *val = tlctx->get_constant(0);
  for (int i = 0; i < (int)stmt->inputs.size(); i++) {
    val = builder->CreateAdd(
        builder->CreateMul(val, tlctx->get_constant(stmt->strides[i])),
        llvm_val[stmt->inputs[i]]);
  }
  llvm_val[stmt] = val;
}

void CodeGenLLVM::visit(IntegerOffsetStmt *stmt) {
  TI_NOT_IMPLEMENTED
  /*
  if (stmt->input->is<GetChStmt>() &&
      stmt->input->as<GetChStmt>()->output_snode->type == SNodeType::place) {
    auto input = stmt->input->as<GetChStmt>();
    auto dtn = input->output_snode->data_type_name();
    emit(R"({}* {}[1] {{({} *)((char *){}[0] + {})}};)", dtn, stmt->raw_name(),
         dtn, stmt->input->raw_name(), stmt->offset);
  } else {
    emit(R"(auto {} = {} + {};)", stmt->raw_name(), stmt->input->raw_name(),
         stmt->offset);
  }
  */
}

void CodeGenLLVM::visit(SNodeLookupStmt *stmt) {
  llvm::Value *parent = nullptr;
  parent = llvm_val[stmt->input_snode];
  TI_ASSERT(parent);
  auto snode = stmt->snode;
  if (snode->type == SNodeType::root) {
    llvm_val[stmt] = builder->CreateGEP(parent, llvm_val[stmt->input_index]);
  } else if (snode->type == SNodeType::dense ||
             snode->type == SNodeType::pointer ||
             snode->type == SNodeType::dynamic ||
             snode->type == SNodeType::bitmasked) {
    if (stmt->activate) {
      call(snode, llvm_val[stmt->input_snode], "activate",
           {llvm_val[stmt->input_index]});
    }
    llvm_val[stmt] = call(snode, llvm_val[stmt->input_snode], "lookup_element",
                          {llvm_val[stmt->input_index]});
  } else {
    TI_INFO(snode_type_name(snode->type));
    TI_NOT_IMPLEMENTED
  }
}

void CodeGenLLVM::visit(GetChStmt *stmt) {
  auto ch = create_call(
      stmt->output_snode->get_ch_from_parent_func_name(),
      {builder->CreateBitCast(llvm_val[stmt->input_ptr],
                              PointerType::getInt8PtrTy(*llvm_context))});
  llvm_val[stmt] = builder->CreateBitCast(
      ch, PointerType::get(StructCompilerLLVM::get_llvm_node_type(
                               module.get(), stmt->output_snode),
                           0));
}

void CodeGenLLVM::visit(ExternalPtrStmt *stmt) {
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
      llvm_val[stmt->base_ptrs[0]],
      llvm::PointerType::get(tlctx->get_data_type(dt), 0));

  auto linear_index = tlctx->get_constant(0);
  for (int i = 0; i < num_indices; i++) {
    linear_index = builder->CreateMul(linear_index, sizes[i]);
    linear_index = builder->CreateAdd(linear_index, llvm_val[stmt->indices[i]]);
  }

  llvm_val[stmt] = builder->CreateGEP(base, linear_index);
}

std::string CodeGenLLVM::init_offloaded_task_function(OffloadedStmt *stmt,
                                                      std::string suffix) {
  current_loop_reentry = nullptr;
  current_while_after_loop = nullptr;
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

void CodeGenLLVM::finalize_offloaded_task_function() {
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

std::tuple<llvm::Value *, llvm::Value *> CodeGenLLVM::get_range_for_bounds(
    OffloadedStmt *stmt) {
  llvm::Value *begin, *end;
  if (stmt->const_begin) {
    begin = tlctx->get_constant(stmt->begin_value);
  } else {
    auto begin_stmt = Stmt::make<GlobalTemporaryStmt>(
        stmt->begin_offset, VectorType(1, DataType::i32));
    begin_stmt->accept(this);
    begin = builder->CreateLoad(llvm_val[begin_stmt.get()]);
  }
  if (stmt->const_end) {
    end = tlctx->get_constant(stmt->end_value);
  } else {
    auto end_stmt = Stmt::make<GlobalTemporaryStmt>(
        stmt->end_offset, VectorType(1, DataType::i32));
    end_stmt->accept(this);
    end = builder->CreateLoad(llvm_val[end_stmt.get()]);
  }
  return std::tuple(begin, end);
}

void CodeGenLLVM::create_offload_struct_for(OffloadedStmt *stmt, bool spmd) {
  llvm::Function *body;
  auto leaf_block = stmt->snode;
  {
    // Create the loop body function
    auto guard = get_function_creation_guard({
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

    if (spmd) {
      threadIdx =
          builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
      blockDim = builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_ntid_x,
                                          {}, {});
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

    // exec_cond: safe-guard the execution of loop body:
    //  - if non-POT tensor dim exists, make sure we don't go out of bounds
    //  - if leaf block is bitmasked, make sure we only loop over active
    //    voxels
    auto exec_cond = tlctx->get_constant(true);
    auto snode = stmt->snode;

    auto coord_object = RuntimeObject("PhysicalCoordinates", this,
                                      builder.get(), new_coordinates);
    for (int i = 0; i < snode->num_active_indices; i++) {
      auto j = snode->physical_index_position[i];
      if (!bit::is_power_of_two(snode->extractors[j].num_elements)) {
        auto coord = coord_object.get("val", tlctx->get_constant(j));
        exec_cond = builder->CreateAnd(
            exec_cond,
            builder->CreateICmp(
                llvm::CmpInst::ICMP_SLT, coord,
                tlctx->get_constant(snode->extractors[j].num_elements)));
      }
    }

    if (snode->type == SNodeType::bitmasked) {
      // test if current voxel is active or not
      auto is_active = call(snode, element.get("element"), "is_active",
                            {builder->CreateLoad(loop_index)});
      is_active =
          builder->CreateTrunc(is_active, llvm::Type::getInt1Ty(*llvm_context));
      exec_cond = builder->CreateAnd(exec_cond, is_active);
    }

    auto body_bb_tail =
        BasicBlock::Create(*llvm_context, "loop_body_tail", func);
    {
      auto bounded_body_bb =
          BasicBlock::Create(*llvm_context, "bound_guarded_loop_body", func);
      builder->CreateCondBr(exec_cond, bounded_body_bb, body_bb_tail);
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
  // TODO: why do we need num_cpu_threads on GPUs?
}

void CodeGenLLVM::visit(LoopIndexStmt *stmt) {
  TI_ASSERT(&module->getContext() == tlctx->get_this_thread_context());
  if (stmt->loop->is<OffloadedStmt>() &&
      stmt->loop->as<OffloadedStmt>()->task_type ==
          OffloadedStmt::TaskType::struct_for) {
    llvm_val[stmt] = builder->CreateLoad(builder->CreateGEP(
        current_coordinates, {tlctx->get_constant(0), tlctx->get_constant(0),
                              tlctx->get_constant(stmt->index)}));
  } else {
    llvm_val[stmt] =
        builder->CreateLoad(loop_vars_llvm[stmt->loop][stmt->index]);
  }
}

void CodeGenLLVM::visit(GlobalTemporaryStmt *stmt) {
  auto runtime = get_runtime();
  auto buffer = call("get_temporary_pointer", runtime,
                     tlctx->get_constant((int64)stmt->offset));

  TI_ASSERT(stmt->width() == 1);
  auto ptr_type =
      llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type.data_type), 0);
  llvm_val[stmt] = builder->CreatePointerCast(buffer, ptr_type);
}

void CodeGenLLVM::visit(InternalFuncStmt *stmt) {
  create_call(stmt->func_name, {get_context()});
}

void CodeGenLLVM::visit(StackAllocaStmt *stmt) {
  TI_ASSERT(stmt->width() == 1);
  auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                   stmt->size_in_bytes());
  auto alloca = create_entry_block_alloca(type, sizeof(int64));
  llvm_val[stmt] = builder->CreateBitCast(
      alloca, llvm::PointerType::getInt8PtrTy(*llvm_context));
  call("stack_init", llvm_val[stmt]);
}

void CodeGenLLVM::visit(StackPopStmt *stmt) {
  call("stack_pop", llvm_val[stmt->stack]);
}

void CodeGenLLVM::visit(StackPushStmt *stmt) {
  auto stack = stmt->stack->as<StackAllocaStmt>();
  call("stack_push", llvm_val[stack], tlctx->get_constant(stack->max_size),
       tlctx->get_constant(stack->element_size_in_bytes()));
  auto primal_ptr = call("stack_top_primal", llvm_val[stack],
                         tlctx->get_constant(stack->element_size_in_bytes()));
  primal_ptr = builder->CreateBitCast(
      primal_ptr, llvm::PointerType::get(
                      tlctx->get_data_type(stmt->ret_type.data_type), 0));
  builder->CreateStore(llvm_val[stmt->v], primal_ptr);
}

void CodeGenLLVM::visit(StackLoadTopStmt *stmt) {
  auto stack = stmt->stack->as<StackAllocaStmt>();
  auto primal_ptr = call("stack_top_primal", llvm_val[stack],
                         tlctx->get_constant(stack->element_size_in_bytes()));
  primal_ptr = builder->CreateBitCast(
      primal_ptr, llvm::PointerType::get(
                      tlctx->get_data_type(stmt->ret_type.data_type), 0));
  llvm_val[stmt] = builder->CreateLoad(primal_ptr);
}

void CodeGenLLVM::visit(StackLoadTopAdjStmt *stmt) {
  auto stack = stmt->stack->as<StackAllocaStmt>();
  auto adjoint = call("stack_top_adjoint", llvm_val[stack],
                      tlctx->get_constant(stack->element_size_in_bytes()));
  adjoint = builder->CreateBitCast(
      adjoint, llvm::PointerType::get(
                   tlctx->get_data_type(stmt->ret_type.data_type), 0));
  llvm_val[stmt] = builder->CreateLoad(adjoint);
}

void CodeGenLLVM::visit(StackAccAdjointStmt *stmt) {
  auto stack = stmt->stack->as<StackAllocaStmt>();
  auto adjoint_ptr = call("stack_top_adjoint", llvm_val[stack],
                          tlctx->get_constant(stack->element_size_in_bytes()));
  adjoint_ptr = builder->CreateBitCast(
      adjoint_ptr, llvm::PointerType::get(
                       tlctx->get_data_type(stack->ret_type.data_type), 0));
  auto old_val = builder->CreateLoad(adjoint_ptr);
  TI_ASSERT(is_real(stmt->v->ret_type.data_type));
  auto new_val = builder->CreateFAdd(old_val, llvm_val[stmt->v]);
  builder->CreateStore(new_val, adjoint_ptr);
}

FunctionType CodeGenLLVM::compile_module_to_executable() {
  TI_AUTO_PROF
  TaichiLLVMContext::eliminate_unused_functions(
      module.get(), [&](std::string func_name) {
        for (auto &task : offloaded_tasks) {
          if (task.name == func_name)
            return true;
        }
        return false;
      });
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

FunctionCreationGuard CodeGenLLVM::get_function_creation_guard(
    std::vector<llvm::Type *> argument_types) {
  return FunctionCreationGuard(this, argument_types);
}

void CodeGenLLVM::initialize_context() {
  if (kernel->arch == Arch::cuda) {
    tlctx = prog->llvm_context_device.get();
  } else {
    tlctx = prog->llvm_context_host.get();
  }
  llvm_context = tlctx->get_this_thread_context();
  builder = std::make_unique<llvm::IRBuilder<>>(*llvm_context);
}

llvm::Value *CodeGenLLVM::get_arg(int i) {
  std::vector<llvm::Value *> args;
  for (auto &arg : func->args()) {
    args.push_back(&arg);
  }
  return args[i];
}

llvm::Value *CodeGenLLVM::get_context() {
  return get_arg(0);
}

llvm::Value *CodeGenLLVM::get_root() {
  return create_call("LLVMRuntime_get_root", {get_runtime()});
}

llvm::Value *CodeGenLLVM::get_runtime() {
  auto runtime_ptr = create_call("Context_get_runtime", {get_context()});
  return builder->CreateBitCast(
      runtime_ptr, llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0));
}

llvm::Value *CodeGenLLVM::emit_struct_meta(SNode *snode) {
  auto obj = emit_struct_meta_object(snode);
  TI_ASSERT(obj != nullptr);
  return obj->ptr;
}

void CodeGenLLVM::emit_to_module() {
  TI_AUTO_PROF
  ir->accept(this);
}

FunctionType CodeGenLLVM::gen() {
  emit_to_module();
  return compile_module_to_executable();
}

TLANG_NAMESPACE_END
