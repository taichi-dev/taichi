#include "taichi/codegen/llvm/codegen_llvm.h"

#include <algorithm>

#ifdef TI_WITH_LLVM
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/program/extension.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/codegen/codegen_utils.h"

namespace taichi::lang {

// TODO: sort function definitions to match declaration order in header

// TODO(k-ye): Hide FunctionCreationGuard inside cpp file
FunctionCreationGuard::FunctionCreationGuard(
    TaskCodeGenLLVM *mb,
    std::vector<llvm::Type *> arguments,
    const std::string &func_name)
    : mb(mb) {
  // Create the loop body function
  auto body_function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*mb->llvm_context), arguments, false);

  body = llvm::Function::Create(body_function_type,
                                llvm::Function::InternalLinkage, func_name,
                                mb->module.get());
  old_func = mb->func;
  // emit into loop body function
  mb->func = body;

  allocas = llvm::BasicBlock::Create(*mb->llvm_context, "allocs", body);
  old_entry = mb->entry_block;
  mb->entry_block = allocas;

  final = llvm::BasicBlock::Create(*mb->llvm_context, "final", body);
  old_final = mb->final_block;
  mb->final_block = final;

  entry = llvm::BasicBlock::Create(*mb->llvm_context, "entry", mb->func);

  ip = mb->builder->saveIP();
  mb->builder->SetInsertPoint(entry);

  auto body_bb =
      llvm::BasicBlock::Create(*mb->llvm_context, "function_body", mb->func);
  mb->builder->CreateBr(body_bb);
  mb->builder->SetInsertPoint(body_bb);
}

FunctionCreationGuard::~FunctionCreationGuard() {
  if (!mb->returned) {
    mb->builder->CreateBr(final);
  }
  mb->builder->SetInsertPoint(final);
  mb->builder->CreateRetVoid();
  mb->returned = false;

  mb->builder->SetInsertPoint(allocas);
  mb->builder->CreateBr(entry);

  mb->entry_block = old_entry;
  mb->final_block = old_final;
  mb->func = old_func;
  mb->builder->restoreIP(ip);

  TI_ASSERT(!llvm::verifyFunction(*body, &llvm::errs()));
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

CodeGenStmtGuard make_loop_reentry_guard(TaskCodeGenLLVM *cg) {
  return CodeGenStmtGuard([cg]() { return cg->current_loop_reentry; },
                          [cg](llvm::BasicBlock *saved_stmt) {
                            cg->current_loop_reentry = saved_stmt;
                          });
}

CodeGenStmtGuard make_while_after_loop_guard(TaskCodeGenLLVM *cg) {
  return CodeGenStmtGuard([cg]() { return cg->current_while_after_loop; },
                          [cg](llvm::BasicBlock *saved_stmt) {
                            cg->current_while_after_loop = saved_stmt;
                          });
}

}  // namespace

// TaskCodeGenLLVM
void TaskCodeGenLLVM::visit(Block *stmt_list) {
  for (auto &stmt : stmt_list->statements) {
    stmt->accept(this);
    if (returned) {
      break;
    }
  }
}

void TaskCodeGenLLVM::visit(AllocaStmt *stmt) {
  auto alloca_type = stmt->ret_type.ptr_removed();
  if (alloca_type->is<TensorType>()) {
    auto tensor_type = alloca_type->cast<TensorType>();
    auto type = tlctx->get_data_type(tensor_type);
    if (stmt->is_shared) {
      auto base = new llvm::GlobalVariable(
          *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
          fmt::format("shared_array_{}", stmt->id), nullptr,
          llvm::GlobalVariable::NotThreadLocal, 3 /*addrspace=shared*/);
      base->setAlignment(llvm::MaybeAlign(8));
      auto ptr_type = llvm::PointerType::get(type, 0);
      llvm_val[stmt] = builder->CreatePointerCast(base, ptr_type);
    } else {
      llvm_val[stmt] = create_entry_block_alloca(type);
    }
  } else {
    llvm_val[stmt] = create_entry_block_alloca(alloca_type);
    // initialize as zero if element is not a pointer
    if (!alloca_type->is<PointerType>())
      builder->CreateStore(tlctx->get_constant(alloca_type, 0), llvm_val[stmt]);
  }
}

void TaskCodeGenLLVM::visit(RandStmt *stmt) {
  if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
    // Promoting to f32 since there's no rand_f16 support in runtime.cpp.
    auto val_f32 = call("rand_f32", get_context());
    llvm_val[stmt] =
        builder->CreateFPTrunc(val_f32, llvm::Type::getHalfTy(*llvm_context));
  } else {
    llvm_val[stmt] = call(
        fmt::format("rand_{}", data_type_name(stmt->ret_type)), get_context());
  }
}

void TaskCodeGenLLVM::emit_extra_unary(UnaryOpStmt *stmt) {
  auto input = llvm_val[stmt->operand];
  auto input_taichi_type = stmt->operand->ret_type;
  if (input_taichi_type->is_primitive(PrimitiveTypeID::f16)) {
    // Promote to f32 since we don't have f16 support for extra unary ops in in
    // runtime.cpp.
    input = builder->CreateFPExt(input, llvm::Type::getFloatTy(*llvm_context));
    input_taichi_type = PrimitiveType::f32;
  }

  auto op = stmt->op_type;
  auto input_type = input->getType();

#define UNARY_STD(x)                                                    \
  else if (op == UnaryOpType::x) {                                      \
    if (input_taichi_type->is_primitive(PrimitiveTypeID::f32)) {        \
      llvm_val[stmt] = call(#x "_f32", input);                          \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::f64)) { \
      llvm_val[stmt] = call(#x "_f64", input);                          \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i32)) { \
      llvm_val[stmt] = call(#x "_i32", input);                          \
    } else if (input_taichi_type->is_primitive(PrimitiveTypeID::i64)) { \
      llvm_val[stmt] = call(#x "_i64", input);                          \
    } else {                                                            \
      TI_NOT_IMPLEMENTED                                                \
    }                                                                   \
  }
  if (false) {
  }
  UNARY_STD(abs)
  UNARY_STD(exp)
  UNARY_STD(log)
  UNARY_STD(tan)
  UNARY_STD(tanh)
  UNARY_STD(sgn)
  UNARY_STD(acos)
  UNARY_STD(asin)
  UNARY_STD(cos)
  UNARY_STD(sin)
  else if (op == UnaryOpType::sqrt) {
    llvm_val[stmt] =
        builder->CreateIntrinsic(llvm::Intrinsic::sqrt, {input_type}, {input});
  }
  else if (op == UnaryOpType::popcnt) {
    llvm_val[stmt] =
        builder->CreateIntrinsic(llvm::Intrinsic::ctpop, {input_type}, {input});
  }
  else {
    TI_P(unary_op_type_name(op));
    TI_NOT_IMPLEMENTED
  }
#undef UNARY_STD
  if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
    // Convert back to f16
    llvm_val[stmt] = builder->CreateFPTrunc(
        llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
  }
}

std::unique_ptr<RuntimeObject> TaskCodeGenLLVM::emit_struct_meta_object(
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
  } else if (snode->type == SNodeType::quant_array) {
    meta = std::make_unique<RuntimeObject>("DenseMeta", this, builder.get());
    emit_struct_meta_base("Dense", meta->ptr, snode);
  } else {
    TI_P(snode_type_name(snode->type));
    TI_NOT_IMPLEMENTED;
  }
  return meta;
}

void TaskCodeGenLLVM::emit_struct_meta_base(const std::string &name,
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
             tlctx->get_constant(snode->max_num_elements()));
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
               get_struct_function(snode->get_ch_from_parent_func_name(),
                                   snode->get_snode_tree_id()));

  if (snode->type != SNodeType::place)
    common.set("refine_coordinates",
               get_struct_function(snode->refine_coordinates_func_name(),
                                   snode->get_snode_tree_id()));
}

TaskCodeGenLLVM::TaskCodeGenLLVM(int id,
                                 const CompileConfig &compile_config,
                                 TaichiLLVMContext &tlctx,
                                 const Kernel *kernel,
                                 IRNode *ir,
                                 std::unique_ptr<llvm::Module> &&module)
    // TODO: simplify LLVMModuleBuilder ctor input
    : LLVMModuleBuilder(
          module == nullptr ? tlctx.new_module("kernel") : std::move(module),
          &tlctx),
      compile_config(compile_config),
      kernel(kernel),
      ir(ir),
      prog(kernel->program),
      task_codegen_id(id) {
  if (ir == nullptr)
    this->ir = kernel->ir.get();
  initialize_context();

  context_ty = get_runtime_type("RuntimeContext");
  physical_coordinate_ty = get_runtime_type(kLLVMPhysicalCoordinatesName);

  kernel_name = kernel->name + "_kernel";
  current_callable = kernel;
}

void TaskCodeGenLLVM::visit(DecorationStmt *stmt) {
}

void TaskCodeGenLLVM::create_elementwise_cast(
    UnaryOpStmt *stmt,
    llvm::Type *to_ty,
    std::function<llvm::Value *(llvm::Value *, llvm::Type *)> func,
    bool on_self) {
  auto from_ty = stmt->operand->ret_type->cast<TensorType>();
  TI_ASSERT_INFO(from_ty,
                 "Cannot perform elementwise ops on non-tensor type {}",
                 from_ty->to_string());
  llvm::Value *vec = llvm::UndefValue::get(llvm::VectorType::get(
      to_ty, from_ty->get_num_elements(), /*scalable=*/false));
  for (int i = 0; i < from_ty->get_num_elements(); ++i) {
    auto elem = builder->CreateExtractElement(
        on_self ? llvm_val[stmt] : llvm_val[stmt->operand], i);
    auto cast_value = func(elem, to_ty);
    vec = builder->CreateInsertElement(vec, cast_value, i);
  }
  llvm_val[stmt] = vec;
}

void TaskCodeGenLLVM::visit(UnaryOpStmt *stmt) {
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
    auto from = stmt->operand->ret_type;
    auto to = stmt->ret_type;
    TI_ASSERT_INFO(
        from->is<TensorType>() == to->is<TensorType>(),
        "Cannot cast between tensor type and non-tensor type: {} v.s. {}",
        from->to_string(), to->to_string());

    if (from == to) {
      llvm_val[stmt] = llvm_val[stmt->operand];
    } else if (to->is_primitive(PrimitiveTypeID::u1)) {
      llvm_val[stmt] = builder->CreateIsNotNull(input);
    } else if (is_real(from.get_element_type()) !=
               is_real(to.get_element_type())) {
      if (is_real(from.get_element_type()) &&
          (is_integral(to.get_element_type()))) {
        cast_op = (is_signed(to.get_element_type()))
                      ? llvm::Instruction::CastOps::FPToSI
                      : llvm::Instruction::CastOps::FPToUI;
      } else if (is_integral(from.get_element_type()) &&
                 is_real(to.get_element_type())) {
        cast_op = (is_signed(from.get_element_type()))
                      ? llvm::Instruction::CastOps::SIToFP
                      : llvm::Instruction::CastOps::UIToFP;
      } else {
        TI_P(data_type_name(from));
        TI_P(data_type_name(to));
        TI_NOT_IMPLEMENTED;
      }
      bool use_f16 = to->is_primitive(PrimitiveTypeID::f16) ||
                     (to->is<TensorType>() &&
                      to->cast<TensorType>()->get_element_type()->is_primitive(
                          PrimitiveTypeID::f16));
      auto cast_type = use_f16 ? (to->is<TensorType>()
                                      ? TypeFactory::create_tensor_type(
                                            to->cast<TensorType>()->get_shape(),
                                            PrimitiveType::f32)
                                      : PrimitiveType::f32)
                               : stmt->cast_type;

      auto cast_func = [this, cast_op](llvm::Value *value, llvm::Type *type) {
        return this->builder->CreateCast(cast_op, value, type);
      };
      if (!cast_type->is<TensorType>()) {
        llvm_val[stmt] = cast_func(input, tlctx->get_data_type(cast_type));
      } else {
        create_elementwise_cast(
            stmt,
            tlctx->get_data_type(
                cast_type->cast<TensorType>()->get_element_type()),
            cast_func);
      }

      if (use_f16) {
        auto trunc_func = [this](llvm::Value *value, llvm::Type *type) {
          return this->builder->CreateFPTrunc(value, type);
        };
        auto to_ty = llvm::Type::getHalfTy(*llvm_context);
        if (!cast_type->is<TensorType>()) {
          llvm_val[stmt] = trunc_func(llvm_val[stmt], to_ty);
        } else {
          create_elementwise_cast(stmt, to_ty, trunc_func, /*on_self=*/true);
        }
      }
    } else if (is_real(from.get_element_type()) &&
               is_real(to.get_element_type())) {
      auto t1 = from->is<TensorType>()
                    ? from->cast<TensorType>()->get_element_type()
                    : from.operator->();
      auto t2 = to->is<TensorType>()
                    ? to->cast<TensorType>()->get_element_type()
                    : to.operator->();
      if (data_type_size(t1) < data_type_size(t2)) {
        auto cast_func = [this](llvm::Value *value, llvm::Type *type) {
          return this->builder->CreateFPExt(value, type);
        };
        if (!stmt->cast_type->is<TensorType>()) {
          llvm_val[stmt] =
              cast_func(input, tlctx->get_data_type(stmt->cast_type));
        } else {
          create_elementwise_cast(
              stmt,
              tlctx->get_data_type(
                  stmt->cast_type->cast<TensorType>()->get_element_type()),
              cast_func);
        }
      } else {
        if (to->is_primitive(PrimitiveTypeID::f16) ||
            (to->is<TensorType>() &&
             to->cast<TensorType>()->get_element_type()->is_primitive(
                 PrimitiveTypeID::f16))) {
          if (!to->is<TensorType>()) {
            llvm_val[stmt] = builder->CreateFPTrunc(
                builder->CreateFPTrunc(llvm_val[stmt->operand],
                                       llvm::Type::getFloatTy(*llvm_context)),
                llvm::Type::getHalfTy(*llvm_context));
          } else {
            auto tensor_type = to->cast<TensorType>();
            llvm::Value *vec = llvm::UndefValue::get(tlctx->get_data_type(to));
            for (int i = 0; i < tensor_type->get_num_elements(); ++i) {
              auto elem = builder->CreateExtractElement(vec, i);
              auto double_trunced = builder->CreateFPTrunc(
                  builder->CreateFPTrunc(elem,
                                         llvm::Type::getFloatTy(*llvm_context)),
                  llvm::Type::getHalfTy(*llvm_context));
              vec = builder->CreateInsertElement(vec, double_trunced, i);
            }
            llvm_val[stmt] = vec;
          }
        } else {
          auto trunc_fn = [this](llvm::Value *value, llvm::Type *type) {
            return this->builder->CreateFPTrunc(value, type);
          };
          auto cast_type =
              stmt->cast_type->is<TensorType>()
                  ? stmt->cast_type->cast<TensorType>()->get_element_type()
                  : stmt->cast_type.operator->();
          if (!stmt->cast_type->is<TensorType>()) {
            llvm_val[stmt] = trunc_fn(input, tlctx->get_data_type(cast_type));
          } else {
            create_elementwise_cast(
                stmt,
                tlctx->get_data_type(
                    cast_type->cast<TensorType>()->get_element_type()),
                trunc_fn);
          }
        }
      }
    } else if (!is_real(from.get_element_type()) &&
               !is_real(to.get_element_type())) {
      llvm_val[stmt] = builder->CreateIntCast(
          llvm_val[stmt->operand], tlctx->get_data_type(to),
          is_signed(from.get_element_type()));
    }
  } else if (stmt->op_type == UnaryOpType::cast_bits) {
    TI_ASSERT(data_type_size(stmt->ret_type) ==
              data_type_size(stmt->cast_type));
    if (stmt->operand->ret_type.is_pointer()) {
      TI_ASSERT(is_integral(stmt->cast_type));
      llvm_val[stmt] = builder->CreatePtrToInt(
          llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
    } else {
      llvm_val[stmt] = builder->CreateBitCast(
          llvm_val[stmt->operand], tlctx->get_data_type(stmt->cast_type));
    }
  } else if (op == UnaryOpType::rsqrt) {
    llvm::Function *sqrt_fn = llvm::Intrinsic::getDeclaration(
        module.get(), llvm::Intrinsic::sqrt, input->getType());
    auto intermediate = builder->CreateCall(sqrt_fn, input, "sqrt");
    llvm_val[stmt] = builder->CreateFDiv(
        tlctx->get_constant(stmt->ret_type, 1.0), intermediate);
  } else if (op == UnaryOpType::bit_not) {
    llvm_val[stmt] = builder->CreateNot(input);
  } else if (op == UnaryOpType::neg) {
    if (is_real(stmt->operand->ret_type)) {
      llvm_val[stmt] = builder->CreateFNeg(input, "neg");
    } else {
      llvm_val[stmt] = builder->CreateNeg(input, "neg");
    }
  } else if (op == UnaryOpType::logic_not) {
    llvm_val[stmt] = builder->CreateIsNull(input);
  }
  UNARY_INTRINSIC(round)
  UNARY_INTRINSIC(floor)
  UNARY_INTRINSIC(ceil)
  else {
    emit_extra_unary(stmt);
  }
#undef UNARY_INTRINSIC
}

void TaskCodeGenLLVM::create_elementwise_binary(
    BinaryOpStmt *stmt,
    std::function<llvm::Value *(llvm::Value *lhs, llvm::Value *rhs)> f) {
  TI_ASSERT(stmt->lhs->ret_type->is<TensorType>());
  TI_ASSERT(stmt->rhs->ret_type->is<TensorType>());
  auto lhs_ty = stmt->lhs->ret_type->cast<TensorType>();
  auto rhs_ty = stmt->rhs->ret_type->cast<TensorType>();
  TI_ASSERT(lhs_ty->get_num_elements() == rhs_ty->get_num_elements());
  auto lhs_vec = llvm_val[stmt->lhs];
  auto rhs_vec = llvm_val[stmt->rhs];
  auto elt_type_name = data_type_name(lhs_ty->get_element_type());
  llvm::Value *result =
      llvm::UndefValue::get(tlctx->get_data_type(stmt->ret_type));
  for (int i = 0; i < lhs_ty->get_num_elements(); ++i) {
    auto lhs = builder->CreateExtractElement(lhs_vec, i);
    auto rhs = builder->CreateExtractElement(rhs_vec, i);
    result = builder->CreateInsertElement(result, f(lhs, rhs), i);
  }
  llvm_val[stmt] = result;
}

void TaskCodeGenLLVM::visit(BinaryOpStmt *stmt) {
  auto op = stmt->op_type;
  auto ret_type = stmt->ret_type;

  if (op == BinaryOpType::add) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateFAdd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#if defined(__clang__) || defined(__GNUC__)
    } else if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] =
          call("debug_add_" + stmt->ret_type->to_string(), get_arg(0),
               llvm_val[stmt->lhs], llvm_val[stmt->rhs],
               builder->CreateGlobalStringPtr(stmt->get_tb()));
#endif
    } else {
      llvm_val[stmt] =
          builder->CreateAdd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::sub) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateFSub(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#if defined(__clang__) || defined(__GNUC__)
    } else if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] =
          call("debug_sub_" + stmt->ret_type->to_string(), get_arg(0),
               llvm_val[stmt->lhs], llvm_val[stmt->rhs],
               builder->CreateGlobalStringPtr(stmt->get_tb()));
#endif
    } else {
      llvm_val[stmt] =
          builder->CreateSub(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::mul) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateFMul(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#if defined(__clang__) || defined(__GNUC__)
    } else if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] =
          call("debug_mul_" + stmt->ret_type->to_string(), get_arg(0),
               llvm_val[stmt->lhs], llvm_val[stmt->rhs],
               builder->CreateGlobalStringPtr(stmt->get_tb()));
#endif
    } else {
      llvm_val[stmt] =
          builder->CreateMul(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::div) {
    if (is_real(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateFDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else if (is_signed(stmt->ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateSDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] =
          builder->CreateUDiv(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::mod) {
    llvm_val[stmt] =
        builder->CreateSRem(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::logical_and) {
    auto *lhs = builder->CreateIsNotNull(llvm_val[stmt->lhs]);
    auto *rhs = builder->CreateIsNotNull(llvm_val[stmt->rhs]);
    llvm_val[stmt] = builder->CreateAnd(lhs, rhs);
    llvm_val[stmt] = builder->CreateZExtOrTrunc(
        llvm_val[stmt], tlctx->get_data_type(stmt->ret_type));
  } else if (op == BinaryOpType::logical_or) {
    auto *lhs = builder->CreateIsNotNull(llvm_val[stmt->lhs]);
    auto *rhs = builder->CreateIsNotNull(llvm_val[stmt->rhs]);
    llvm_val[stmt] = builder->CreateOr(lhs, rhs);
    llvm_val[stmt] = builder->CreateZExtOrTrunc(
        llvm_val[stmt], tlctx->get_data_type(stmt->ret_type));
  } else if (op == BinaryOpType::bit_and) {
    llvm_val[stmt] =
        builder->CreateAnd(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_or) {
    llvm_val[stmt] =
        builder->CreateOr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_xor) {
    llvm_val[stmt] =
        builder->CreateXor(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
  } else if (op == BinaryOpType::bit_shl) {
#if defined(__clang__) || defined(__GNUC__)
    if (compile_config.debug && is_integral(stmt->ret_type)) {
      llvm_val[stmt] =
          call("debug_shl_" + stmt->ret_type->to_string(), get_arg(0),
               llvm_val[stmt->lhs], llvm_val[stmt->rhs],
               builder->CreateGlobalStringPtr(stmt->get_tb()));
    } else {
      llvm_val[stmt] =
          builder->CreateShl(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
#else
    llvm_val[stmt] =
        builder->CreateShl(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
#endif
  } else if (op == BinaryOpType::bit_sar) {
    if (is_signed(stmt->lhs->ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateAShr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    } else {
      llvm_val[stmt] =
          builder->CreateLShr(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
  } else if (op == BinaryOpType::max) {
#define BINARYOP_MAX(x)                                            \
  else if (ret_type->is_primitive(PrimitiveTypeID::x)) {           \
    llvm_val[stmt] =                                               \
        call("max_" #x, llvm_val[stmt->lhs], llvm_val[stmt->rhs]); \
  }

    if (is_real(ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateMaxNum(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
    BINARYOP_MAX(u16)
    BINARYOP_MAX(i16)
    BINARYOP_MAX(u32)
    BINARYOP_MAX(i32)
    BINARYOP_MAX(u64)
    BINARYOP_MAX(i64)
    else {
      if (auto tensor_ty = ret_type->cast<TensorType>()) {
        auto elt_ty = tensor_ty->get_element_type();
        TI_ASSERT(elt_ty->is_primitive(PrimitiveTypeID::u16) ||
                  elt_ty->is_primitive(PrimitiveTypeID::i16) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u32) ||
                  elt_ty->is_primitive(PrimitiveTypeID::i32) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u64) ||
                  elt_ty->is_primitive(PrimitiveTypeID::i64));
        auto dtype_name = data_type_name(elt_ty);
        auto binary_max = [this, &dtype_name](llvm::Value *lhs,
                                              llvm::Value *rhs) {
          return call("max_" + dtype_name, lhs, rhs);
        };
        create_elementwise_binary(stmt, binary_max);
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    }
  } else if (op == BinaryOpType::min) {
#define BINARYOP_MIN(x)                                            \
  else if (ret_type->is_primitive(PrimitiveTypeID::x)) {           \
    llvm_val[stmt] =                                               \
        call("min_" #x, llvm_val[stmt->lhs], llvm_val[stmt->rhs]); \
  }

    if (is_real(ret_type.get_element_type())) {
      llvm_val[stmt] =
          builder->CreateMinNum(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
    }
    BINARYOP_MIN(u16)
    BINARYOP_MIN(i16)
    BINARYOP_MIN(u32)
    BINARYOP_MIN(i32)
    BINARYOP_MIN(u64)
    BINARYOP_MIN(i64)
    else {
      if (auto tensor_ty = ret_type->cast<TensorType>()) {
        auto elt_ty = tensor_ty->get_element_type();
        TI_ASSERT(elt_ty->is_primitive(PrimitiveTypeID::u16) ||
                  elt_ty->is_primitive(PrimitiveTypeID::i16) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u32) ||
                  elt_ty->is_primitive(PrimitiveTypeID::i32) ||
                  elt_ty->is_primitive(PrimitiveTypeID::u64) ||
                  elt_ty->is_primitive(PrimitiveTypeID::i64));
        auto dtype_name = data_type_name(elt_ty);
        auto binary_min = [this, &dtype_name](llvm::Value *lhs,
                                              llvm::Value *rhs) {
          return call("min_" + dtype_name, lhs, rhs);
        };
        create_elementwise_binary(stmt, binary_min);
      } else {
        TI_P(data_type_name(ret_type));
        TI_NOT_IMPLEMENTED
      }
    }
  } else if (is_comparison(op)) {
    llvm::Value *cmp = nullptr;
    auto input_type = stmt->lhs->ret_type;
    if (op == BinaryOpType::cmp_eq) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOEQ(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        cmp = builder->CreateICmpEQ(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      }
    } else if (op == BinaryOpType::cmp_le) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOLE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp =
              builder->CreateICmpSLE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpULE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_ge) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp =
              builder->CreateICmpSGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpUGE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_lt) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOLT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp =
              builder->CreateICmpSLT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpULT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_gt) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpOGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        if (is_signed(input_type.get_element_type())) {
          cmp =
              builder->CreateICmpSGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        } else {
          cmp =
              builder->CreateICmpUGT(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
        }
      }
    } else if (op == BinaryOpType::cmp_ne) {
      if (is_real(input_type.get_element_type())) {
        cmp = builder->CreateFCmpONE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      } else {
        cmp = builder->CreateICmpNE(llvm_val[stmt->lhs], llvm_val[stmt->rhs]);
      }
    } else {
      TI_NOT_IMPLEMENTED
    }
    llvm_val[stmt] = cmp;
  } else {
    // This branch contains atan2 and pow which use runtime.cpp function for
    // **real** type. We don't have f16 support there so promoting to f32 is
    // necessary.
    llvm::Value *lhs = llvm_val[stmt->lhs];
    llvm::Value *rhs = llvm_val[stmt->rhs];
    if (stmt->lhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      lhs = builder->CreateFPExt(lhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (stmt->rhs->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      rhs = builder->CreateFPExt(rhs, llvm::Type::getFloatTy(*llvm_context));
    }
    if (ret_type->is_primitive(PrimitiveTypeID::f16)) {
      ret_type = PrimitiveType::f32;
    }

    if (op == BinaryOpType::atan2) {
      if (arch_is_cpu(current_arch())) {
        if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
          llvm_val[stmt] = call("atan2_f32", lhs, rhs);
        } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
          llvm_val[stmt] = call("atan2_f64", lhs, rhs);
        } else {
          TI_P(data_type_name(ret_type));
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else if (op == BinaryOpType::pow) {
      if (arch_is_cpu(current_arch())) {
        // Note that ret_type here cannot be integral because pow with an
        // integral exponent has been demoted in the demote_operations pass
        if (ret_type->is_primitive(PrimitiveTypeID::f32)) {
          llvm_val[stmt] = call("pow_f32", lhs, rhs);
        } else if (ret_type->is_primitive(PrimitiveTypeID::f64)) {
          llvm_val[stmt] = call("pow_f64", lhs, rhs);
        } else {
          TI_P(data_type_name(ret_type));
          TI_NOT_IMPLEMENTED
        }
      } else {
        TI_NOT_IMPLEMENTED
      }
    } else {
      TI_P(binary_op_type_name(op));
      TI_NOT_IMPLEMENTED
    }

    // Convert back to f16 if applicable.
    if (stmt->ret_type->is_primitive(PrimitiveTypeID::f16)) {
      llvm_val[stmt] = builder->CreateFPTrunc(
          llvm_val[stmt], llvm::Type::getHalfTy(*llvm_context));
    }
  }
}

void TaskCodeGenLLVM::visit(TernaryOpStmt *stmt) {
  TI_ASSERT(stmt->op_type == TernaryOpType::select);
  llvm_val[stmt] =
      builder->CreateSelect(builder->CreateIsNotNull(llvm_val[stmt->op1]),
                            llvm_val[stmt->op2], llvm_val[stmt->op3]);
}

void TaskCodeGenLLVM::visit(IfStmt *if_stmt) {
  // TODO: take care of vectorized cases
  llvm::BasicBlock *true_block =
      llvm::BasicBlock::Create(*llvm_context, "true_block", func);
  llvm::BasicBlock *false_block =
      llvm::BasicBlock::Create(*llvm_context, "false_block", func);
  llvm::BasicBlock *after_if =
      llvm::BasicBlock::Create(*llvm_context, "after_if", func);
  llvm::Value *cond = builder->CreateIsNotNull(llvm_val[if_stmt->cond]);
  builder->CreateCondBr(cond, true_block, false_block);
  builder->SetInsertPoint(true_block);
  if (if_stmt->true_statements) {
    if_stmt->true_statements->accept(this);
  }
  if (!returned) {
    builder->CreateBr(after_if);
  } else {
    returned = false;
  }
  builder->SetInsertPoint(false_block);
  if (if_stmt->false_statements) {
    if_stmt->false_statements->accept(this);
  }
  if (!returned) {
    builder->CreateBr(after_if);
  } else {
    returned = false;
  }
  builder->SetInsertPoint(after_if);
}

llvm::Value *TaskCodeGenLLVM::create_print(std::string tag,
                                           DataType dt,
                                           llvm::Value *value) {
  if (!arch_is_cpu(compile_config.arch)) {
    TI_WARN("print not supported on arch {}", arch_name(compile_config.arch));
    return nullptr;
  }
  std::vector<llvm::Value *> args;
  std::string format = data_type_format(dt);
  auto runtime_printf = call("LLVMRuntime_get_host_printf", get_runtime());
  args.push_back(builder->CreateGlobalStringPtr(
      ("[llvm codegen debug] " + tag + " = " + format + "\n").c_str(),
      "format_string"));
  if (dt->is_primitive(PrimitiveTypeID::f32))
    value =
        builder->CreateFPExt(value, tlctx->get_data_type(PrimitiveType::f64));
  args.push_back(value);

  auto func_type_func = get_runtime_function("get_func_type_host_printf");
  return call(runtime_printf, func_type_func->getFunctionType(),
              std::move(args));
}

llvm::Value *TaskCodeGenLLVM::create_print(std::string tag,
                                           llvm::Value *value) {
  if (value->getType() == llvm::Type::getFloatTy(*llvm_context))
    return create_print(
        tag,
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32),
        value);
  else if (value->getType() == llvm::Type::getInt32Ty(*llvm_context))
    return create_print(
        tag,
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32),
        value);
  else if (value->getType() == llvm::Type::getHalfTy(*llvm_context)) {
    auto extended =
        builder->CreateFPExt(value, llvm::Type::getFloatTy(*llvm_context));
    return create_print(
        tag,
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::f32),
        extended);
  } else if (value->getType() == llvm::Type::getInt64Ty(*llvm_context))
    return create_print(
        tag,
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i64),
        value);
  else if (value->getType() == llvm::Type::getInt16Ty(*llvm_context))
    return create_print(
        tag,
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i16),
        value);
  else
    TI_NOT_IMPLEMENTED
}

void TaskCodeGenLLVM::visit(PrintStmt *stmt) {
  std::vector<llvm::Value *> args;
  std::string formats;
  auto value_for_printf = [this](llvm::Value *to_print, DataType dtype) {
    if (dtype->is_primitive(PrimitiveTypeID::f32) ||
        dtype->is_primitive(PrimitiveTypeID::f16))
      return this->builder->CreateFPExt(
          to_print, this->tlctx->get_data_type(PrimitiveType::f64));
    if (dtype->is_primitive(PrimitiveTypeID::i8))
      return builder->CreateSExt(to_print,
                                 tlctx->get_data_type(PrimitiveType::i16));
    if (dtype->is_primitive(PrimitiveTypeID::u8))
      return builder->CreateZExt(to_print,
                                 tlctx->get_data_type(PrimitiveType::u16));
    if (dtype->is_primitive(PrimitiveTypeID::u1))
      return builder->CreateZExt(to_print,
                                 tlctx->get_data_type(PrimitiveType::i32));
    return to_print;
  };
  for (auto i = 0; i < stmt->contents.size(); ++i) {
    auto const &content = stmt->contents[i];
    auto const &format = stmt->formats[i];

    if (std::holds_alternative<Stmt *>(content)) {
      auto arg_stmt = std::get<Stmt *>(content);
      auto value = llvm_val[arg_stmt];
      auto value_type = value->getType();
      if (arg_stmt->ret_type->is<TensorType>()) {
        auto dtype = arg_stmt->ret_type->cast<TensorType>();
        auto elem_type = dtype->get_element_type();
        for (int i = 0; i < dtype->get_num_elements(); ++i) {
          if (codegen_vector_type(compile_config)) {
            TI_ASSERT(llvm::dyn_cast<llvm::VectorType>(value_type));
            auto elem = builder->CreateExtractElement(value, i);
            args.push_back(value_for_printf(elem, elem_type));
          } else {
            TI_ASSERT(llvm::dyn_cast<llvm::ArrayType>(value_type));
            auto elem = builder->CreateExtractValue(value, i);
            args.push_back(value_for_printf(elem, elem_type));
          }
        }
        formats += data_type_format(arg_stmt->ret_type);
      } else {
        args.push_back(value_for_printf(value, arg_stmt->ret_type));
        formats += merge_printf_specifier(format,
                                          data_type_format(arg_stmt->ret_type));
      }
    } else {
      auto arg_str = std::get<std::string>(content);
      auto value = builder->CreateGlobalStringPtr(arg_str, "content_string");
      args.push_back(value);
      formats += "%s";
    }
  }
  auto runtime_printf = call("LLVMRuntime_get_host_printf", get_runtime());
  args.insert(args.begin(),
              builder->CreateGlobalStringPtr(formats.c_str(), "format_string"));
  auto func_type_func = get_runtime_function("get_func_type_host_printf");
  llvm_val[stmt] =
      call(runtime_printf, func_type_func->getFunctionType(), std::move(args));
}

void TaskCodeGenLLVM::visit(ConstStmt *stmt) {
  auto val = stmt->val;
  if (val.dt->is_primitive(PrimitiveTypeID::f32)) {
    llvm_val[stmt] =
        llvm::ConstantFP::get(*llvm_context, llvm::APFloat(val.val_float32()));
  } else if (val.dt->is_primitive(PrimitiveTypeID::f16)) {
    llvm_val[stmt] = llvm::ConstantFP::get(llvm::Type::getHalfTy(*llvm_context),
                                           val.val_float16());
  } else if (val.dt->is_primitive(PrimitiveTypeID::f64)) {
    llvm_val[stmt] =
        llvm::ConstantFP::get(*llvm_context, llvm::APFloat(val.val_float64()));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u1)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(1, (uint64)val.val_uint1(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i8)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(8, (uint64)val.val_int8(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u8)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(8, (uint64)val.val_uint8(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i16)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(16, (uint64)val.val_int16(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u16)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(16, (uint64)val.val_uint16(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i32)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(32, (uint64)val.val_int32(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u32)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(32, (uint64)val.val_uint32(), false));
  } else if (val.dt->is_primitive(PrimitiveTypeID::i64)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(64, (uint64)val.val_int64(), true));
  } else if (val.dt->is_primitive(PrimitiveTypeID::u64)) {
    llvm_val[stmt] = llvm::ConstantInt::get(
        *llvm_context, llvm::APInt(64, val.val_uint64(), false));
  } else {
    TI_P(data_type_name(val.dt));
    TI_NOT_IMPLEMENTED;
  }
}

void TaskCodeGenLLVM::visit(WhileControlStmt *stmt) {
  using namespace llvm;

  BasicBlock *after_break =
      BasicBlock::Create(*llvm_context, "after_break", func);
  TI_ASSERT(current_while_after_loop);
  auto *cond = builder->CreateIsNull(llvm_val[stmt->cond]);
  builder->CreateCondBr(cond, current_while_after_loop, after_break);
  builder->SetInsertPoint(after_break);
}

void TaskCodeGenLLVM::visit(ContinueStmt *stmt) {
  using namespace llvm;
  auto stmt_in_off_range_for = [stmt]() {
    TI_ASSERT(stmt->scope != nullptr);
    if (auto *offl = stmt->scope->cast<OffloadedStmt>(); offl) {
      TI_ASSERT(offl->task_type == OffloadedStmt::TaskType::range_for ||
                offl->task_type == OffloadedStmt::TaskType::struct_for);
      return offl->task_type == OffloadedStmt::TaskType::range_for;
    }
    return false;
  };
  if (stmt_in_off_range_for()) {
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

void TaskCodeGenLLVM::visit(WhileStmt *stmt) {
  using namespace llvm;
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

  if (!returned) {
    builder->CreateBr(body);  // jump to head
  } else {
    returned = false;
  }

  builder->SetInsertPoint(after_loop);
}

llvm::Value *TaskCodeGenLLVM::cast_pointer(llvm::Value *val,
                                           std::string dest_ty_name,
                                           int addr_space) {
  return builder->CreateBitCast(
      val, llvm::PointerType::get(get_runtime_type(dest_ty_name), addr_space));
}

void TaskCodeGenLLVM::emit_list_gen(OffloadedStmt *listgen) {
  auto snode_child = listgen->snode;
  auto snode_parent = listgen->snode->parent;
  auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
  auto meta_parent = cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
  if (snode_parent->type == SNodeType::root) {
    // Since there's only one container to expand, we need a special kernel for
    // more parallelism.
    call("element_listgen_root", get_runtime(), meta_parent, meta_child);
  } else {
    call("element_listgen_nonroot", get_runtime(), meta_parent, meta_child);
  }
}

void TaskCodeGenLLVM::emit_gc(OffloadedStmt *stmt) {
  auto snode = stmt->snode->id;
  call("node_gc", get_runtime(), tlctx->get_constant(snode));
}

void TaskCodeGenLLVM::create_increment(llvm::Value *ptr, llvm::Value *value) {
  auto original_value = builder->CreateLoad(value->getType(), ptr);
  builder->CreateStore(builder->CreateAdd(original_value, value), ptr);
}

void TaskCodeGenLLVM::create_naive_range_for(RangeForStmt *for_stmt) {
  using namespace llvm;
  BasicBlock *body = BasicBlock::Create(*llvm_context, "for_loop_body", func);
  BasicBlock *loop_inc =
      BasicBlock::Create(*llvm_context, "for_loop_inc", func);
  BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "after_for", func);
  BasicBlock *loop_test =
      BasicBlock::Create(*llvm_context, "for_loop_test", func);

  auto loop_var_ty = tlctx->get_data_type(PrimitiveType::i32);

  auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
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
                                 builder->CreateLoad(loop_var_ty, loop_var),
                                 llvm_val[for_stmt->end]);
    } else {
      cond = builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SGE,
                                 builder->CreateLoad(loop_var_ty, loop_var),
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
    if (!returned) {
      builder->CreateBr(loop_inc);
    } else {
      returned = false;
    }
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

void TaskCodeGenLLVM::visit(RangeForStmt *for_stmt) {
  create_naive_range_for(for_stmt);
}

llvm::Value *TaskCodeGenLLVM::bitcast_from_u64(llvm::Value *val,
                                               DataType type) {
  llvm::Type *dest_ty = nullptr;
  TI_ASSERT(!type->is<PointerType>());
  if (auto qit = type->cast<QuantIntType>()) {
    if (qit->get_is_signed())
      dest_ty = tlctx->get_data_type(PrimitiveType::i32);
    else
      dest_ty = tlctx->get_data_type(PrimitiveType::u32);
  } else {
    dest_ty = tlctx->get_data_type(type);
  }
  auto dest_bits = dest_ty->getPrimitiveSizeInBits();
  if (dest_ty == llvm::Type::getHalfTy(*llvm_context)) {
    // if dest_ty == half, CreateTrunc will only keep low 16bits of mantissa
    // which doesn't mean anything.
    // So we truncate to 32 bits first and then fptrunc to half if applicable
    auto truncated =
        builder->CreateTrunc(val, llvm::Type::getIntNTy(*llvm_context, 32));
    auto casted = builder->CreateBitCast(truncated,
                                         llvm::Type::getFloatTy(*llvm_context));
    return builder->CreateFPTrunc(casted, llvm::Type::getHalfTy(*llvm_context));
  } else {
    auto truncated = builder->CreateTrunc(
        val, llvm::Type::getIntNTy(*llvm_context, dest_bits));

    return builder->CreateBitCast(truncated, dest_ty);
  }
}

llvm::Value *TaskCodeGenLLVM::bitcast_to_u64(llvm::Value *val, DataType type) {
  auto intermediate_bits = 0;
  if (type.is_pointer()) {
    return builder->CreatePtrToInt(val, tlctx->get_data_type<int64>());
  }
  if (auto qit = type->cast<QuantIntType>()) {
    intermediate_bits = data_type_bits(qit->get_compute_type());
  } else {
    intermediate_bits = tlctx->get_data_type(type)->getPrimitiveSizeInBits();
  }
  llvm::Type *dest_ty = tlctx->get_data_type<int64>();
  llvm::Type *intermediate_type = nullptr;
  if (val->getType() == llvm::Type::getHalfTy(*llvm_context)) {
    val = builder->CreateFPExt(val, tlctx->get_data_type<float>());
    intermediate_type = tlctx->get_data_type<int32>();
  } else {
    intermediate_type = llvm::Type::getIntNTy(*llvm_context, intermediate_bits);
  }
  return builder->CreateZExt(builder->CreateBitCast(val, intermediate_type),
                             dest_ty);
}

void TaskCodeGenLLVM::visit(ArgLoadStmt *stmt) {
  if (stmt->arg_depth > 0) {
    llvm_val[stmt] =
        get_argpack_arg(stmt->arg_id, stmt->arg_depth, stmt->create_load);
  } else {
    llvm_val[stmt] = get_struct_arg(stmt->arg_id, stmt->create_load);
  }
}

void TaskCodeGenLLVM::visit(ReturnStmt *stmt) {
  auto types = stmt->element_types();
  if (std::any_of(types.begin(), types.end(),
                  [](const DataType &t) { return t.is_pointer(); })) {
    TI_NOT_IMPLEMENTED
  } else {
    TI_ASSERT(stmt->values.size() ==
              current_callable->ret_type->get_flattened_num_elements());
    auto *buffer = call("RuntimeContext_get_result_buffer", get_context());
    set_struct_to_buffer(current_callable->ret_type, buffer, stmt->values);
  }
  builder->CreateBr(final_block);
  returned = true;
}

void TaskCodeGenLLVM::visit(LocalLoadStmt *stmt) {
  // FIXME: get ptr_ty from taichi instead of llvm.
  llvm::Type *ptr_ty = nullptr;
  auto *val = llvm_val[stmt->src];
  if (auto *alloc = llvm::dyn_cast<llvm::AllocaInst>(val))
    ptr_ty = alloc->getAllocatedType();
  if (!ptr_ty && stmt->src->element_type().is_pointer()) {
    ptr_ty = tlctx->get_data_type(stmt->src->element_type().ptr_removed());
  }
  TI_ASSERT(ptr_ty);
  llvm_val[stmt] = builder->CreateLoad(ptr_ty, llvm_val[stmt->src]);
}

void TaskCodeGenLLVM::visit(LocalStoreStmt *stmt) {
  builder->CreateStore(llvm_val[stmt->val], llvm_val[stmt->dest]);
}

void TaskCodeGenLLVM::visit(AssertStmt *stmt) {
  TI_ASSERT((int)stmt->args.size() <= taichi_error_message_max_num_arguments);
  auto argument_buffer_size = llvm::ArrayType::get(
      llvm::Type::getInt64Ty(*llvm_context), stmt->args.size());

  // TODO: maybe let all asserts in a single offload share a single buffer?
  auto arguments = create_entry_block_alloca(argument_buffer_size);

  std::vector<llvm::Value *> args;
  args.emplace_back(get_runtime());
  args.emplace_back(builder->CreateIsNotNull(llvm_val[stmt->cond]));
  args.emplace_back(builder->CreateGlobalStringPtr(stmt->text));

  for (int i = 0; i < stmt->args.size(); i++) {
    auto arg = stmt->args[i];
    TI_ASSERT(llvm_val[arg]);

    // First convert the argument to an integral type with the same number of
    // bits:
    auto cast_type = llvm::Type::getIntNTy(
        *llvm_context, 8 * (std::size_t)data_type_size(arg->ret_type));
    auto cast_int = builder->CreateBitCast(llvm_val[arg], cast_type);

    // Then zero-extend the conversion result into int64:
    auto cast_int64 =
        builder->CreateZExt(cast_int, llvm::Type::getInt64Ty(*llvm_context));

    // Finally store the int64 value to the argument buffer:
    builder->CreateStore(
        cast_int64,
        builder->CreateGEP(argument_buffer_size, arguments,
                           {tlctx->get_constant(0), tlctx->get_constant(i)}));
  }

  args.emplace_back(tlctx->get_constant((int)stmt->args.size()));
  args.emplace_back(
      builder->CreateGEP(argument_buffer_size, arguments,
                         {tlctx->get_constant(0), tlctx->get_constant(0)}));

  llvm_val[stmt] = call("taichi_assert_format", std::move(args));
}

void TaskCodeGenLLVM::visit(SNodeOpStmt *stmt) {
  auto snode = stmt->snode;
  if (stmt->op_type == SNodeOpType::allocate) {
    TI_ASSERT(snode->type == SNodeType::dynamic);
    TI_ASSERT(stmt->ret_type.is_pointer() &&
              stmt->ret_type.ptr_removed()->is_primitive(PrimitiveTypeID::gen));
    auto ptr =
        call(snode, llvm_val[stmt->ptr], "allocate", {llvm_val[stmt->val]});
    llvm_val[stmt] = ptr;
  } else if (stmt->op_type == SNodeOpType::length) {
    TI_ASSERT(snode->type == SNodeType::dynamic);
    llvm_val[stmt] = call(snode, llvm_val[stmt->ptr], "get_num_elements", {});
  } else if (stmt->op_type == SNodeOpType::is_active) {
    llvm_val[stmt] =
        call(snode, llvm_val[stmt->ptr], "is_active", {llvm_val[stmt->val]});
  } else if (stmt->op_type == SNodeOpType::activate) {
    llvm_val[stmt] =
        call(snode, llvm_val[stmt->ptr], "activate", {llvm_val[stmt->val]});
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

llvm::Value *TaskCodeGenLLVM::optimized_reduction(AtomicOpStmt *stmt) {
  return nullptr;
}

llvm::Value *TaskCodeGenLLVM::quant_type_atomic(AtomicOpStmt *stmt) {
  // TODO(type): support all AtomicOpTypes on quant types
  if (stmt->op_type != AtomicOpType::add) {
    return nullptr;
  }

  auto dst_type = stmt->dest->ret_type->as<PointerType>()->get_pointee_type();
  if (auto qit = dst_type->cast<QuantIntType>()) {
    return atomic_add_quant_int(
        llvm_val[stmt->dest],
        tlctx->get_data_type(
            stmt->dest->as<GetChStmt>()->input_snode->physical_type),
        qit, llvm_val[stmt->val], is_signed(stmt->val->ret_type));
  } else if (auto qfxt = dst_type->cast<QuantFixedType>()) {
    return atomic_add_quant_fixed(
        llvm_val[stmt->dest],
        tlctx->get_data_type(
            stmt->dest->as<GetChStmt>()->input_snode->physical_type),
        qfxt, llvm_val[stmt->val]);
  } else {
    return nullptr;
  }
}

llvm::Value *TaskCodeGenLLVM::integral_type_atomic(AtomicOpStmt *stmt) {
  if (!is_integral(stmt->val->ret_type)) {
    return nullptr;
  }

  // Atomic operators not supported by LLVM, we implement them using CAS
  if (stmt->op_type == AtomicOpType::mul) {
    return atomic_op_using_cas(
        llvm_val[stmt->dest], llvm_val[stmt->val],
        [&](auto v1, auto v2) { return builder->CreateMul(v1, v2); },
        stmt->val->ret_type);
  }
  // Atomic operators supported by LLVM
  std::unordered_map<AtomicOpType, llvm::AtomicRMWInst::BinOp> bin_op;
  bin_op[AtomicOpType::add] = llvm::AtomicRMWInst::BinOp::Add;
  if (is_signed(stmt->val->ret_type)) {
    bin_op[AtomicOpType::min] = llvm::AtomicRMWInst::BinOp::Min;
    bin_op[AtomicOpType::max] = llvm::AtomicRMWInst::BinOp::Max;
  } else {
    bin_op[AtomicOpType::min] = llvm::AtomicRMWInst::BinOp::UMin;
    bin_op[AtomicOpType::max] = llvm::AtomicRMWInst::BinOp::UMax;
  }
  bin_op[AtomicOpType::bit_and] = llvm::AtomicRMWInst::BinOp::And;
  bin_op[AtomicOpType::bit_or] = llvm::AtomicRMWInst::BinOp::Or;
  bin_op[AtomicOpType::bit_xor] = llvm::AtomicRMWInst::BinOp::Xor;
  TI_ASSERT(bin_op.find(stmt->op_type) != bin_op.end());
  return builder->CreateAtomicRMW(
      bin_op.at(stmt->op_type), llvm_val[stmt->dest], llvm_val[stmt->val],
      llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent);
}

llvm::Value *TaskCodeGenLLVM::atomic_op_using_cas(
    llvm::Value *dest,
    llvm::Value *val,
    std::function<llvm::Value *(llvm::Value *, llvm::Value *)> op,
    const DataType &type) {
  using namespace llvm;
  BasicBlock *body = BasicBlock::Create(*llvm_context, "while_loop_body", func);
  BasicBlock *after_loop =
      BasicBlock::Create(*llvm_context, "after_while", func);

  builder->CreateBr(body);
  builder->SetInsertPoint(body);

  llvm::Value *old_val;

  {
    int bits = data_type_bits(type);
    llvm::PointerType *typeIntPtr = get_integer_ptr_type(bits);
    llvm::IntegerType *typeIntTy = get_integer_type(bits);

    old_val = builder->CreateLoad(val->getType(), dest);
    auto new_val = op(old_val, val);
    dest = builder->CreateBitCast(dest, typeIntPtr);
    auto atomicCmpXchg = builder->CreateAtomicCmpXchg(
        dest, builder->CreateBitCast(old_val, typeIntTy),
        builder->CreateBitCast(new_val, typeIntTy), llvm::MaybeAlign(0),
        AtomicOrdering::SequentiallyConsistent,
        AtomicOrdering::SequentiallyConsistent);
    // Check whether CAS was succussful
    auto ok = builder->CreateExtractValue(atomicCmpXchg, 1);
    builder->CreateCondBr(builder->CreateNot(ok), body, after_loop);
  }

  builder->SetInsertPoint(after_loop);

  return old_val;
}

llvm::Value *TaskCodeGenLLVM::real_type_atomic(AtomicOpStmt *stmt) {
  if (!is_real(stmt->val->ret_type)) {
    return nullptr;
  }

  PrimitiveTypeID prim_type = stmt->val->ret_type->cast<PrimitiveType>()->type;
  AtomicOpType op = stmt->op_type;
  if (prim_type == PrimitiveTypeID::f16) {
    switch (op) {
      case AtomicOpType::add:
        return atomic_op_using_cas(
            llvm_val[stmt->dest], llvm_val[stmt->val],
            [&](auto v1, auto v2) { return builder->CreateFAdd(v1, v2); },
            stmt->val->ret_type);
      case AtomicOpType::max:
        return atomic_op_using_cas(
            llvm_val[stmt->dest], llvm_val[stmt->val],
            [&](auto v1, auto v2) { return builder->CreateMaxNum(v1, v2); },
            stmt->val->ret_type);
      case AtomicOpType::min:
        return atomic_op_using_cas(
            llvm_val[stmt->dest], llvm_val[stmt->val],
            [&](auto v1, auto v2) { return builder->CreateMinNum(v1, v2); },
            stmt->val->ret_type);
      default:
        break;
    }
  }

  switch (op) {
    case AtomicOpType::add:
      return builder->CreateAtomicRMW(
          llvm::AtomicRMWInst::FAdd, llvm_val[stmt->dest], llvm_val[stmt->val],
          llvm::MaybeAlign(0), llvm::AtomicOrdering::SequentiallyConsistent);
    case AtomicOpType::mul:
      return atomic_op_using_cas(
          llvm_val[stmt->dest], llvm_val[stmt->val],
          [&](auto v1, auto v2) { return builder->CreateFMul(v1, v2); },
          stmt->val->ret_type);
    default:
      break;
  }

  std::unordered_map<PrimitiveTypeID,
                     std::unordered_map<AtomicOpType, std::string>>
      atomics;
  atomics[PrimitiveTypeID::f32][AtomicOpType::min] = "atomic_min_f32";
  atomics[PrimitiveTypeID::f64][AtomicOpType::min] = "atomic_min_f64";
  atomics[PrimitiveTypeID::f32][AtomicOpType::max] = "atomic_max_f32";
  atomics[PrimitiveTypeID::f64][AtomicOpType::max] = "atomic_max_f64";
  TI_ASSERT(atomics.find(prim_type) != atomics.end());
  TI_ASSERT(atomics.at(prim_type).find(op) != atomics.at(prim_type).end());
  return call(atomics.at(prim_type).at(op), llvm_val[stmt->dest],
              llvm_val[stmt->val]);
}

void TaskCodeGenLLVM::visit(AtomicOpStmt *stmt) {
  bool is_local = stmt->dest->is<AllocaStmt>();
  if (is_local) {
    TI_ERROR("Local atomics should have been demoted.");
  }
  llvm::Value *old_value;
  if (llvm::Value *result = optimized_reduction(stmt)) {
    old_value = result;
  } else if (llvm::Value *result = quant_type_atomic(stmt)) {
    old_value = result;
  } else if (llvm::Value *result = real_type_atomic(stmt)) {
    old_value = result;
  } else if (llvm::Value *result = integral_type_atomic(stmt)) {
    old_value = result;
  } else {
    TI_NOT_IMPLEMENTED
  }
  llvm_val[stmt] = old_value;
}

void TaskCodeGenLLVM::visit(GlobalPtrStmt *stmt) {
  TI_ERROR("Global Ptrs should have been lowered.");
}

void TaskCodeGenLLVM::visit(GlobalStoreStmt *stmt) {
  TI_ASSERT(llvm_val[stmt->val]);
  TI_ASSERT(llvm_val[stmt->dest]);
  auto ptr_type = stmt->dest->ret_type->as<PointerType>();
  if (ptr_type->is_bit_pointer()) {
    auto pointee_type = ptr_type->get_pointee_type();
    auto snode = stmt->dest->as<GetChStmt>()->input_snode;
    if (snode->type == SNodeType::bit_struct) {
      TI_ERROR(
          "Bit struct stores with type {} should have been handled by "
          "BitStructStoreStmt.",
          pointee_type->to_string());
    }
    if (auto qit = pointee_type->cast<QuantIntType>()) {
      store_quant_int(llvm_val[stmt->dest],
                      tlctx->get_data_type(snode->physical_type), qit,
                      llvm_val[stmt->val], true);
    } else if (auto qfxt = pointee_type->cast<QuantFixedType>()) {
      store_quant_fixed(llvm_val[stmt->dest],
                        tlctx->get_data_type(snode->physical_type), qfxt,
                        llvm_val[stmt->val], true);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  } else {
    builder->CreateStore(llvm_val[stmt->val], llvm_val[stmt->dest]);
  }
}

llvm::Value *TaskCodeGenLLVM::create_intrinsic_load(llvm::Value *ptr,
                                                    llvm::Type *ty) {
  TI_NOT_IMPLEMENTED;
}

void TaskCodeGenLLVM::create_global_load(GlobalLoadStmt *stmt,
                                         bool should_cache_as_read_only) {
  auto ptr = llvm_val[stmt->src];
  auto ptr_type = stmt->src->ret_type->as<PointerType>();
  if (ptr_type->is_bit_pointer()) {
    auto val_type = ptr_type->get_pointee_type();
    auto get_ch = stmt->src->as<GetChStmt>();
    auto physical_type =
        tlctx->get_data_type(get_ch->input_snode->physical_type);
    auto [byte_ptr, bit_offset] = load_bit_ptr(ptr);
    auto physical_value = should_cache_as_read_only
                              ? create_intrinsic_load(byte_ptr, physical_type)
                              : builder->CreateLoad(physical_type, byte_ptr);
    if (auto qit = val_type->cast<QuantIntType>()) {
      llvm_val[stmt] = extract_quant_int(physical_value, bit_offset, qit);
    } else if (auto qfxt = val_type->cast<QuantFixedType>()) {
      qit = qfxt->get_digits_type()->as<QuantIntType>();
      auto digits = extract_quant_int(physical_value, bit_offset, qit);
      llvm_val[stmt] = reconstruct_quant_fixed(digits, qfxt);
    } else {
      TI_ASSERT(val_type->is<QuantFloatType>());
      TI_ASSERT(get_ch->input_snode->dt->is<BitStructType>());
      llvm_val[stmt] = extract_quant_float(
          physical_value, get_ch->input_snode->dt->as<BitStructType>(),
          get_ch->output_snode->id_in_bit_struct);
    }
  } else {
    // Byte pointer case.
    if (should_cache_as_read_only) {
      llvm_val[stmt] =
          create_intrinsic_load(ptr, tlctx->get_data_type(stmt->ret_type));
    } else {
      llvm_val[stmt] =
          builder->CreateLoad(tlctx->get_data_type(stmt->ret_type), ptr);
    }
  }
}

void TaskCodeGenLLVM::visit(GlobalLoadStmt *stmt) {
  create_global_load(stmt, false);
}

std::string TaskCodeGenLLVM::get_runtime_snode_name(SNode *snode) {
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
  } else if (snode->type == SNodeType::bit_struct) {
    return "BitStruct";
  } else if (snode->type == SNodeType::quant_array) {
    return "QuantArray";
  } else {
    TI_P(snode_type_name(snode->type));
    TI_NOT_IMPLEMENTED
  }
}

llvm::Value *TaskCodeGenLLVM::call(
    SNode *snode,
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

  return call(prefix + "_" + method, std::move(func_arguments));
}

llvm::Function *TaskCodeGenLLVM::get_struct_function(const std::string &name,
                                                     int tree_id) {
  used_tree_ids.insert(tree_id);
  auto f = tlctx->get_struct_function(name, tree_id);
  if (!f) {
    TI_ERROR("Struct function {} not found.", name);
  }
  f = llvm::cast<llvm::Function>(
      module
          ->getOrInsertFunction(name, f->getFunctionType(), f->getAttributes())
          .getCallee());
  return f;
}

template <typename... Args>
llvm::Value *TaskCodeGenLLVM::call_struct_func(int tree_id,
                                               const std::string &func_name,
                                               Args &&...args) {
  auto func = get_struct_function(func_name, tree_id);
  auto arglist = std::vector<llvm::Value *>({args...});
  check_func_call_signature(func->getFunctionType(), func->getName(), arglist,
                            builder.get());
  return builder->CreateCall(func, arglist);
}

void TaskCodeGenLLVM::visit(GetRootStmt *stmt) {
  if (stmt->root() == nullptr)
    llvm_val[stmt] = builder->CreateBitCast(
        get_root(SNodeTree::kFirstID),
        llvm::PointerType::get(
            StructCompilerLLVM::get_llvm_node_type(
                module.get(), prog->get_snode_root(SNodeTree::kFirstID)),
            0));
  else
    llvm_val[stmt] = builder->CreateBitCast(
        get_root(stmt->root()->get_snode_tree_id()),
        llvm::PointerType::get(
            StructCompilerLLVM::get_llvm_node_type(module.get(), stmt->root()),
            0));
}

void TaskCodeGenLLVM::visit(LinearizeStmt *stmt) {
  llvm::Value *val = tlctx->get_constant(0);
  for (int i = 0; i < (int)stmt->inputs.size(); i++) {
    val = builder->CreateAdd(
        builder->CreateMul(val, tlctx->get_constant(stmt->strides[i])),
        llvm_val[stmt->inputs[i]]);
  }
  llvm_val[stmt] = val;
}

void TaskCodeGenLLVM::visit(IntegerOffsetStmt *stmt){TI_NOT_IMPLEMENTED}

llvm::Value *TaskCodeGenLLVM::create_bit_ptr(llvm::Value *byte_ptr,
                                             llvm::Value *bit_offset) {
  // 1. define the bit pointer struct (X=8/16/32/64)
  // struct bit_pointer_X {
  //    iX* byte_ptr;
  //    i32 bit_offset;
  // };
  TI_ASSERT(bit_offset->getType()->isIntegerTy(32));
  auto struct_type = llvm::StructType::get(
      *llvm_context, {byte_ptr->getType(), bit_offset->getType()});
  // 2. allocate the bit pointer struct
  auto bit_ptr = create_entry_block_alloca(struct_type);
  // 3. store `byte_ptr`
  builder->CreateStore(byte_ptr, builder->CreateGEP(struct_type, bit_ptr,
                                                    {tlctx->get_constant(0),
                                                     tlctx->get_constant(0)}));
  // 4. store `bit_offset
  builder->CreateStore(
      bit_offset,
      builder->CreateGEP(struct_type, bit_ptr,
                         {tlctx->get_constant(0), tlctx->get_constant(1)}));
  return bit_ptr;
}

std::tuple<llvm::Value *, llvm::Value *> TaskCodeGenLLVM::load_bit_ptr(
    llvm::Value *bit_ptr) {
  // FIXME: get ptr_ty from taichi instead of llvm.
  llvm::Type *ptr_ty = nullptr;
  if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(bit_ptr))
    ptr_ty = AI->getAllocatedType();
  TI_ASSERT(ptr_ty);
  auto *struct_ty = llvm::cast<llvm::StructType>(ptr_ty);
  auto byte_ptr = builder->CreateLoad(
      struct_ty->getElementType(0),
      builder->CreateGEP(ptr_ty, bit_ptr,
                         {tlctx->get_constant(0), tlctx->get_constant(0)}));
  auto bit_offset = builder->CreateLoad(
      struct_ty->getElementType(1),
      builder->CreateGEP(ptr_ty, bit_ptr,
                         {tlctx->get_constant(0), tlctx->get_constant(1)}));

  return std::make_tuple(byte_ptr, bit_offset);
}

void TaskCodeGenLLVM::visit(SNodeLookupStmt *stmt) {
  llvm::Value *parent = nullptr;
  parent = llvm_val[stmt->input_snode];
  TI_ASSERT(parent);
  auto snode = stmt->snode;
  if (snode->type == SNodeType::root) {
    // FIXME: get parent_type from taichi instead of llvm.
    llvm::Type *parent_ty = builder->getInt8Ty();
    if (auto bit_cast = llvm::dyn_cast<llvm::BitCastInst>(parent)) {
      parent_ty = bit_cast->getDestTy();
      if (auto ptr_ty = llvm::dyn_cast<llvm::PointerType>(parent_ty))
        parent_ty = ptr_ty->getPointerElementType();
    }
    llvm_val[stmt] =
        builder->CreateGEP(parent_ty, parent, llvm_val[stmt->input_index]);
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
  } else if (snode->type == SNodeType::bit_struct) {
    llvm_val[stmt] = parent;
  } else if (snode->type == SNodeType::quant_array) {
    auto element_num_bits =
        snode->dt->as<QuantArrayType>()->get_element_num_bits();
    auto offset = tlctx->get_constant(element_num_bits);
    offset = builder->CreateMul(offset, llvm_val[stmt->input_index]);
    llvm_val[stmt] = create_bit_ptr(llvm_val[stmt->input_snode], offset);
  } else {
    TI_INFO(snode_type_name(snode->type));
    TI_NOT_IMPLEMENTED
  }
}

void TaskCodeGenLLVM::visit(GetChStmt *stmt) {
  if (stmt->input_snode->type == SNodeType::quant_array) {
    llvm_val[stmt] = llvm_val[stmt->input_ptr];
  } else if (stmt->ret_type->as<PointerType>()->is_bit_pointer()) {
    auto bit_struct = stmt->input_snode->dt->cast<BitStructType>();
    auto bit_offset =
        bit_struct->get_member_bit_offset(stmt->output_snode->id_in_bit_struct);
    auto offset = tlctx->get_constant(bit_offset);
    llvm_val[stmt] = create_bit_ptr(llvm_val[stmt->input_ptr], offset);
  } else {
    auto ch = call_struct_func(
        stmt->output_snode->get_snode_tree_id(),
        stmt->output_snode->get_ch_from_parent_func_name(),
        builder->CreateBitCast(llvm_val[stmt->input_ptr],
                               llvm::PointerType::getInt8PtrTy(*llvm_context)));
    llvm_val[stmt] = builder->CreateBitCast(
        ch, llvm::PointerType::get(StructCompilerLLVM::get_llvm_node_type(
                                       module.get(), stmt->output_snode),
                                   0));
  }
}

void TaskCodeGenLLVM::visit(MatrixPtrStmt *stmt) {
  if (stmt->offset_used_as_index()) {
    auto type = tlctx->get_data_type(stmt->origin->ret_type.ptr_removed());

    auto casted_ptr = builder->CreateBitCast(llvm_val[stmt->origin],
                                             llvm::PointerType::get(type, 0));

    llvm_val[stmt] = builder->CreateGEP(
        type, casted_ptr, {tlctx->get_constant(0), llvm_val[stmt->offset]});
  } else {
    // Access PtrOffset via: base_ptr + offset
    auto origin_address = builder->CreatePtrToInt(
        llvm_val[stmt->origin], llvm::Type::getInt64Ty(*llvm_context));
    auto address_offset = builder->CreateSExt(
        llvm_val[stmt->offset], llvm::Type::getInt64Ty(*llvm_context));
    auto target_address = builder->CreateAdd(origin_address, address_offset);
    auto dt = stmt->ret_type.ptr_removed();
    llvm_val[stmt] = builder->CreateIntToPtr(
        target_address, llvm::PointerType::get(tlctx->get_data_type(dt), 0));
  }
}

void TaskCodeGenLLVM::visit(ExternalPtrStmt *stmt) {
  // Index into ndarray struct
  DataType operand_dtype = stmt->base_ptr->ret_type.ptr_removed()
                               ->as<StructType>()
                               ->get_element_type({1})
                               ->as<PointerType>()
                               ->get_pointee_type();
  auto arg_type = operand_dtype;
  if (operand_dtype->is<TensorType>()) {
    arg_type = operand_dtype->as<TensorType>()->get_element_type();
  }
  auto ptr_type = TypeFactory::get_instance().get_pointer_type(arg_type);
  auto members =
      stmt->base_ptr->ret_type.ptr_removed()->as<StructType>()->elements();
  bool needs_grad = members.size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
  auto struct_type =
      tlctx->get_data_type(TypeFactory::get_instance().get_ndarray_struct_type(
          arg_type, stmt->ndim, needs_grad));
  auto *gep = builder->CreateGEP(
      struct_type, llvm_val.at(stmt->base_ptr),
      {tlctx->get_constant(0), tlctx->get_constant(int(stmt->is_grad) + 1)});
  auto *ptr_val = builder->CreateLoad(tlctx->get_data_type(ptr_type), gep);

  int num_indices = stmt->indices.size();
  std::vector<llvm::Value *> sizes(num_indices);
  auto dt = stmt->ret_type.ptr_removed();
  int num_element_indices =
      dt->is<TensorType>() ? 0 : stmt->element_shape.size();

  /*
    ExternalPtrStmt can be divided into "outter" and "inner" parts.

    For example, "x" is an Ndarray with shape = (5, 5, 6), m=2, n=3.
    Indexing to a single element of "x" is of form: x[i, j, k][m, n]

    The "outter" part is x[i, j, k], and the "inner" part is [m, n].
    Shape of the inner part is known at compile time, stored in its ret_type.
    Shape of the outter part is determined at runtime, passed from the
    "extra_args".

    "num_indices - num_element_indices" gives how many "extra_args" to read from
  */
  int num_array_args = num_indices - num_element_indices;
  const size_t element_shape_index_offset = num_array_args;

  for (int i = 0; i < num_array_args; i++) {
    auto raw_arg = builder->CreateGEP(
        struct_type, llvm_val[stmt->base_ptr],
        {tlctx->get_constant(0),
         tlctx->get_constant(TypeFactory::SHAPE_POS_IN_NDARRAY),
         tlctx->get_constant(i)});
    raw_arg =
        builder->CreateLoad(tlctx->get_data_type(PrimitiveType::i32), raw_arg);
    sizes[i] = raw_arg;
  }

  auto linear_index = tlctx->get_constant(0);
  size_t size_var_index = 0;
  for (int i = 0; i < num_indices; i++) {
    if (i >= element_shape_index_offset &&
        i < element_shape_index_offset + num_element_indices) {
      // Indexing TensorType-elements
      llvm::Value *size_var = tlctx->get_constant(
          stmt->element_shape[i - element_shape_index_offset]);
      linear_index = builder->CreateMul(linear_index, size_var);
    } else {
      // Indexing array dimensions
      linear_index = builder->CreateMul(linear_index, sizes[size_var_index++]);
    }
    linear_index = builder->CreateAdd(linear_index, llvm_val[stmt->indices[i]]);
  }
  TI_ASSERT(size_var_index == num_indices - num_element_indices);

  /*
    llvm::GEP implicitly indicates alignment when used upon llvm::VectorType.
    For example:

      "getelementptr <10 x i32>* %1, 0, 1" is interpreted as "%1 + 16(aligned)"

    However, this does not fit with Taichi's Ndarray semantics. We will have to
    do pointer arithmetics to manually calculate the offset.
  */
  if (operand_dtype->is<TensorType>()) {
    // Access PtrOffset via: base_ptr + offset * sizeof(element)

    auto address_offset = builder->CreateSExt(
        linear_index, llvm::Type::getInt64Ty(*llvm_context));

    auto stmt_ret_type = stmt->ret_type.ptr_removed();
    if (stmt_ret_type->is<TensorType>()) {
      // This case corresponds to outter indexing only
      // The stride for linear_index is num_elements() in TensorType.
      address_offset = builder->CreateMul(
          address_offset,
          tlctx->get_constant(
              get_data_type<int64>(),
              stmt_ret_type->cast<TensorType>()->get_num_elements()));
    } else {
      // This case corresponds to outter + inner indexing
      // Since both outter and inner indices are linearized into linear_index,
      // the stride for linear_index is 1, and there's nothing to do here.
    }

    auto ret_ptr = builder->CreateGEP(tlctx->get_data_type(arg_type), ptr_val,
                                      address_offset);
    llvm_val[stmt] = builder->CreateBitCast(
        ret_ptr, llvm::PointerType::get(tlctx->get_data_type(dt), 0));

  } else {
    auto base_ty = tlctx->get_data_type(dt);
    auto base =
        builder->CreateBitCast(ptr_val, llvm::PointerType::get(base_ty, 0));

    llvm_val[stmt] = builder->CreateGEP(base_ty, base, linear_index);
  }
}

void TaskCodeGenLLVM::visit(ExternalTensorShapeAlongAxisStmt *stmt) {
  const auto arg_id = stmt->arg_id;
  const auto axis = stmt->axis;
  auto extended_arg_id = arg_id;
  extended_arg_id.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
  extended_arg_id.push_back(axis);
  llvm_val[stmt] = get_struct_arg(extended_arg_id, /*create_load=*/true);
}

void TaskCodeGenLLVM::visit(ExternalTensorBasePtrStmt *stmt) {
  auto arg_id = stmt->arg_id;
  int pos = stmt->is_grad ? TypeFactory::GRAD_PTR_POS_IN_NDARRAY
                          : TypeFactory::DATA_PTR_POS_IN_NDARRAY;
  arg_id.push_back(pos);
  llvm_val[stmt] = get_struct_arg(arg_id, /*create_load=*/true);
}

std::string TaskCodeGenLLVM::init_offloaded_task_function(OffloadedStmt *stmt,
                                                          std::string suffix) {
  current_loop_reentry = nullptr;
  current_while_after_loop = nullptr;

  task_function_type =
      llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                              {llvm::PointerType::get(context_ty, 0)}, false);

  auto task_kernel_name = fmt::format(
      "{}_{}_{}{}", kernel_name, task_codegen_id, stmt->task_name(), suffix);
  func = llvm::Function::Create(task_function_type,
                                llvm::Function::ExternalLinkage,
                                task_kernel_name, module.get());

  current_task = std::make_unique<OffloadedTask>(task_kernel_name);

  for (auto &arg : func->args()) {
    kernel_args.push_back(&arg);
  }
  kernel_args[0]->setName("context");
  if (kernel_argument_by_val())
    func->addParamAttr(
        0, llvm::Attribute::getWithByValType(*llvm_context, context_ty));
  // entry_block has all the allocas
  this->entry_block = llvm::BasicBlock::Create(*llvm_context, "entry", func);
  this->final_block = llvm::BasicBlock::Create(*llvm_context, "final", func);

  // The real function body
  func_body_bb = llvm::BasicBlock::Create(*llvm_context, "body", func);
  builder->SetInsertPoint(func_body_bb);
  return task_kernel_name;
}

void TaskCodeGenLLVM::finalize_offloaded_task_function() {
  if (!returned) {
    builder->CreateBr(final_block);
  } else {
    returned = false;
  }
  builder->SetInsertPoint(final_block);
  builder->CreateRetVoid();

  // entry_block should jump to the body after all allocas are inserted
  builder->SetInsertPoint(entry_block);
  builder->CreateBr(func_body_bb);

  if (compile_config.print_kernel_llvm_ir) {
    static FileSequenceWriter writer("taichi_kernel_generic_llvm_ir_{:04d}.ll",
                                     "unoptimized LLVM IR (generic)");
    writer.write(module.get());
  }
  TI_ASSERT(!llvm::verifyFunction(*func, &llvm::errs()));
  // TI_INFO("Kernel function verified.");
}

std::tuple<llvm::Value *, llvm::Value *> TaskCodeGenLLVM::get_range_for_bounds(
    OffloadedStmt *stmt) {
  llvm::Value *begin, *end;
  if (stmt->const_begin) {
    begin = tlctx->get_constant(stmt->begin_value);
  } else {
    auto begin_stmt =
        Stmt::make<GlobalTemporaryStmt>(stmt->begin_offset, PrimitiveType::i32);
    begin_stmt->accept(this);
    begin = builder->CreateLoad(tlctx->get_data_type(PrimitiveType::i32),
                                llvm_val[begin_stmt.get()]);
  }
  if (stmt->const_end) {
    end = tlctx->get_constant(stmt->end_value);
  } else {
    auto end_stmt =
        Stmt::make<GlobalTemporaryStmt>(stmt->end_offset, PrimitiveType::i32);
    end_stmt->accept(this);
    end = builder->CreateLoad(tlctx->get_data_type(PrimitiveType::i32),
                              llvm_val[end_stmt.get()]);
  }
  return std::tuple(begin, end);
}

void TaskCodeGenLLVM::create_offload_struct_for(OffloadedStmt *stmt) {
  using namespace llvm;
  // TODO: instead of constructing tons of LLVM IR, writing the logic in
  // runtime.cpp may be a cleaner solution. See
  // TaskCodeGenCPU::create_offload_range_for as an example.

  llvm::Function *body = nullptr;
  auto leaf_block = stmt->snode;

  // For a bit-vectorized loop over a quant array, we generate struct for on its
  // parent node (must be "dense") instead of itself for higher performance.
  if (stmt->is_bit_vectorized) {
    if (leaf_block->type == SNodeType::quant_array &&
        leaf_block->parent->type == SNodeType::dense) {
      leaf_block = leaf_block->parent;
    } else {
      TI_ERROR(
          "A bit-vectorized struct-for must loop over a quant array with a "
          "dense parent");
    }
  }

  {
    // Create the loop body function
    auto guard = get_function_creation_guard({
        llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
        get_tls_buffer_type(),
        llvm::PointerType::get(get_runtime_type("Element"), 0),
        tlctx->get_data_type<int>(),
        tlctx->get_data_type<int>(),
    });

    body = guard.body;

    /* Function structure:
     *
     * function_body (entry):
     *   loop_index = lower_bound;
     *   tls_prologue()
     *   bls_prologue()
     *   goto loop_test
     *
     * loop_test:
     *   if (loop_index < upper_bound)
     *     goto loop_body
     *   else
     *     goto func_exit
     *
     * loop_body:
     *   initialize_coordinates()
     *   if (bitmasked voxel is active)
     *     goto struct_for_body
     *   else
     *     goto loop_body_tail
     *
     * struct_for_body:
     *   ... (Run codegen on the StructForStmt::body Taichi Block)
     *   goto loop_body_tail
     *
     * loop_body_tail:
     *   loop_index += block_dim
     *   goto loop_test
     *
     * func_exit:
     *   bls_epilogue()
     *   tls_epilogue()
     *   return
     */
    auto loop_index_ty = llvm::Type::getInt32Ty(*llvm_context);
    auto loop_index = create_entry_block_alloca(loop_index_ty);

    RuntimeObject element("Element", this, builder.get(), get_arg(2));

    // Loop ranges
    auto lower_bound = get_arg(3);
    auto upper_bound = get_arg(4);

    parent_coordinates = element.get_ptr("pcoord");
    block_corner_coordinates =
        create_entry_block_alloca(physical_coordinate_ty);

    auto refine =
        get_struct_function(leaf_block->refine_coordinates_func_name(),
                            leaf_block->get_snode_tree_id());
    // A block corner is the global coordinate/index of the lower-left corner
    // cell within that block, and is the same for all the cells within that
    // block.
    call(refine, parent_coordinates, block_corner_coordinates,
         tlctx->get_constant(0));

    if (stmt->tls_prologue) {
      stmt->tls_prologue->accept(this);
    }

    if (stmt->bls_prologue) {
      call("block_barrier");  // "__syncthreads()"
      stmt->bls_prologue->accept(this);
      call("block_barrier");  // "__syncthreads()"
    }

    auto [thread_idx, block_dim] = this->get_spmd_info();
    builder->CreateStore(builder->CreateAdd(thread_idx, lower_bound),
                         loop_index);

    auto loop_test_bb = BasicBlock::Create(*llvm_context, "loop_test", func);
    auto loop_body_bb = BasicBlock::Create(*llvm_context, "loop_body", func);
    auto body_tail_bb =
        BasicBlock::Create(*llvm_context, "loop_body_tail", func);
    auto func_exit = BasicBlock::Create(*llvm_context, "func_exit", func);
    auto struct_for_body_bb =
        BasicBlock::Create(*llvm_context, "struct_for_body_body", func);

    auto lrg = make_loop_reentry_guard(this);
    current_loop_reentry = body_tail_bb;

    builder->CreateBr(loop_test_bb);

    {
      // loop_test:
      //   if (loop_index < upper_bound)
      //     goto loop_body;
      //   else
      //     goto func_exit

      builder->SetInsertPoint(loop_test_bb);
      auto cond = builder->CreateICmp(
          llvm::CmpInst::Predicate::ICMP_SLT,
          builder->CreateLoad(loop_index_ty, loop_index), upper_bound);
      builder->CreateCondBr(cond, loop_body_bb, func_exit);
    }

    // ***********************
    // Begin loop_body_bb:
    builder->SetInsertPoint(loop_body_bb);

    // initialize the coordinates
    auto new_coordinates = create_entry_block_alloca(physical_coordinate_ty);

    call(refine, parent_coordinates, new_coordinates,
         builder->CreateLoad(loop_index_ty, loop_index));

    // For a bit-vectorized loop over a quant array, one more refine step is
    // needed to make final coordinates non-consecutive, since each thread will
    // process multiple coordinates via vectorization
    if (stmt->is_bit_vectorized) {
      refine = get_struct_function(stmt->snode->refine_coordinates_func_name(),
                                   stmt->snode->get_snode_tree_id());
      call(refine, new_coordinates, new_coordinates, tlctx->get_constant(0));
    }

    current_coordinates = new_coordinates;

    // exec_cond: safe-guard the execution of loop body:
    //  - if non-POT field dim exists, make sure we don't go out of bounds
    //  - if leaf block is bitmasked, make sure we only loop over active
    //    voxels
    auto exec_cond = tlctx->get_constant(true);
    auto coord_object = RuntimeObject(kLLVMPhysicalCoordinatesName, this,
                                      builder.get(), new_coordinates);

    if (leaf_block->type == SNodeType::bitmasked ||
        leaf_block->type == SNodeType::pointer) {
      // test whether the current voxel is active or not
      auto is_active = call(leaf_block, element.get("element"), "is_active",
                            {builder->CreateLoad(loop_index_ty, loop_index)});
      is_active = builder->CreateIsNotNull(is_active);
      exec_cond = builder->CreateAnd(exec_cond, is_active);
    }

    builder->CreateCondBr(exec_cond, struct_for_body_bb, body_tail_bb);

    {
      builder->SetInsertPoint(struct_for_body_bb);

      // The real loop body of the StructForStmt
      stmt->body->accept(this);

      builder->CreateBr(body_tail_bb);
    }

    {
      // body tail: increment loop_index and jump to loop_test
      builder->SetInsertPoint(body_tail_bb);

      create_increment(loop_index, block_dim);
      builder->CreateBr(loop_test_bb);

      builder->SetInsertPoint(func_exit);
    }

    if (stmt->bls_epilogue) {
      call("block_barrier");  // "__syncthreads()"
      stmt->bls_epilogue->accept(this);
      call("block_barrier");  // "__syncthreads()"
    }

    if (stmt->tls_epilogue) {
      stmt->tls_epilogue->accept(this);
    }
  }

  int list_element_size = std::min(leaf_block->max_num_elements(),
                                   (int64)taichi_listgen_max_element_size);
  int num_splits = std::max(1, list_element_size / stmt->block_dim +
                                   (list_element_size % stmt->block_dim != 0));

  auto struct_for_func = get_runtime_function("parallel_struct_for");

  if (arch_is_gpu(current_arch())) {
    struct_for_func = llvm::cast<llvm::Function>(
        module
            ->getOrInsertFunction(
                tlctx->get_struct_for_func_name(stmt->tls_size),
                struct_for_func->getFunctionType(),
                struct_for_func->getAttributes())
            .getCallee());
    struct_for_tls_sizes.insert(stmt->tls_size);
  }
  // Loop over nodes in the element list, in parallel
  call(struct_for_func, get_context(), tlctx->get_constant(leaf_block->id),
       tlctx->get_constant(list_element_size), tlctx->get_constant(num_splits),
       body, tlctx->get_constant(stmt->tls_size),
       tlctx->get_constant(stmt->num_cpu_threads));
  // TODO: why do we need num_cpu_threads on GPUs?

  current_coordinates = nullptr;
  parent_coordinates = nullptr;
  block_corner_coordinates = nullptr;
}

void TaskCodeGenLLVM::visit(LoopIndexStmt *stmt) {
  if (stmt->loop->is<OffloadedStmt>() &&
      stmt->loop->as<OffloadedStmt>()->task_type ==
          OffloadedStmt::TaskType::struct_for) {
    llvm::Type *struct_ty = nullptr;
    // FIXME: get struct_ty from taichi instead of llvm.
    if (auto *alloca = llvm::dyn_cast<llvm::AllocaInst>(current_coordinates)) {
      struct_ty = alloca->getAllocatedType();
    }
    TI_ASSERT(struct_ty);
    auto *GEP =
        builder->CreateGEP(struct_ty, current_coordinates,
                           {tlctx->get_constant(0), tlctx->get_constant(0),
                            tlctx->get_constant(stmt->index)});
    if (stmt->index == 0 && !llvm::isa<llvm::GEPOperator>(GEP))
      GEP = builder->CreateBitCast(GEP, struct_ty->getPointerTo());
    llvm_val[stmt] =
        builder->CreateLoad(llvm::Type::getInt32Ty(*llvm_context), GEP);
  } else {
    llvm_val[stmt] =
        builder->CreateLoad(llvm::Type::getInt32Ty(*llvm_context),
                            loop_vars_llvm[stmt->loop][stmt->index]);
  }
}

void TaskCodeGenLLVM::visit(LoopLinearIndexStmt *stmt) {
  if (stmt->loop->is<OffloadedStmt>() &&
      (stmt->loop->as<OffloadedStmt>()->task_type ==
           OffloadedStmt::TaskType::struct_for ||
       stmt->loop->as<OffloadedStmt>()->task_type ==
           OffloadedStmt::TaskType::mesh_for)) {
    llvm_val[stmt] = call("thread_idx");
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

void TaskCodeGenLLVM::visit(BlockCornerIndexStmt *stmt) {
  if (stmt->loop->is<OffloadedStmt>() &&
      stmt->loop->as<OffloadedStmt>()->task_type ==
          OffloadedStmt::TaskType::struct_for) {
    TI_ASSERT(block_corner_coordinates);
    // Make sure physical_coordinate_ty matches
    // struct PhysicalCoordinates {
    //   i32 val[taichi_max_num_indices];
    // };
    TI_ASSERT(physical_coordinate_ty->isStructTy());
    auto physical_coordinate_ty_as_struct =
        llvm::cast<llvm::StructType>(physical_coordinate_ty);
    TI_ASSERT(physical_coordinate_ty_as_struct);
    TI_ASSERT(physical_coordinate_ty_as_struct->getNumElements() == 1);
    auto val_ty = physical_coordinate_ty_as_struct->getElementType(0);
    TI_ASSERT(val_ty->isArrayTy());
    auto val_ty_as_array = llvm::cast<llvm::ArrayType>(val_ty);
    llvm_val[stmt] = builder->CreateLoad(
        val_ty_as_array->getElementType(),
        builder->CreateGEP(physical_coordinate_ty, block_corner_coordinates,
                           {tlctx->get_constant(0), tlctx->get_constant(0),
                            tlctx->get_constant(stmt->index)}));
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

void TaskCodeGenLLVM::visit(GlobalTemporaryStmt *stmt) {
  auto runtime = get_runtime();
  auto buffer = call("get_temporary_pointer", runtime,
                     tlctx->get_constant((int64)stmt->offset));

  auto ptr_type = llvm::PointerType::get(
      tlctx->get_data_type(stmt->ret_type.ptr_removed()), 0);
  llvm_val[stmt] = builder->CreatePointerCast(buffer, ptr_type);
}

void TaskCodeGenLLVM::visit(ThreadLocalPtrStmt *stmt) {
  auto base = get_tls_base_ptr();
  auto ptr = builder->CreateGEP(llvm::Type::getInt8Ty(*llvm_context), base,
                                tlctx->get_constant(stmt->offset));
  auto ptr_type = llvm::PointerType::get(
      tlctx->get_data_type(stmt->ret_type.ptr_removed()), 0);
  llvm_val[stmt] = builder->CreatePointerCast(ptr, ptr_type);
}

void TaskCodeGenLLVM::visit(BlockLocalPtrStmt *stmt) {
  TI_ASSERT(bls_buffer);
  auto base = bls_buffer;
  auto ptr =
      builder->CreateGEP(base->getValueType(), base,
                         {tlctx->get_constant(0), llvm_val[stmt->offset]});
  auto ptr_type = llvm::PointerType::get(
      tlctx->get_data_type(stmt->ret_type.ptr_removed()), 0);
  llvm_val[stmt] = builder->CreatePointerCast(ptr, ptr_type);
}

void TaskCodeGenLLVM::visit(ClearListStmt *stmt) {
  auto snode_child = stmt->snode;
  auto snode_parent = stmt->snode->parent;
  auto meta_child = cast_pointer(emit_struct_meta(snode_child), "StructMeta");
  auto meta_parent = cast_pointer(emit_struct_meta(snode_parent), "StructMeta");
  call("clear_list", get_runtime(), meta_parent, meta_child);
}

void TaskCodeGenLLVM::visit(InternalFuncStmt *stmt) {
  std::vector<llvm::Value *> args;

  if (stmt->with_runtime_context)
    args.push_back(get_context());

  for (auto s : stmt->args) {
    args.push_back(llvm_val[s]);
  }
  llvm_val[stmt] = call(stmt->func_name, std::move(args));
}

void TaskCodeGenLLVM::visit(AdStackAllocaStmt *stmt) {
  TI_ASSERT_INFO(stmt->max_size > 0,
                 "Adaptive autodiff stack's size should have been determined.");
  auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                   stmt->size_in_bytes());
  auto alloca = create_entry_block_alloca(type, sizeof(int64));
  llvm_val[stmt] = builder->CreateBitCast(
      alloca, llvm::PointerType::getInt8PtrTy(*llvm_context));
  call("stack_init", llvm_val[stmt]);
}

void TaskCodeGenLLVM::visit(AdStackPopStmt *stmt) {
  call("stack_pop", llvm_val[stmt->stack]);
}

void TaskCodeGenLLVM::visit(AdStackPushStmt *stmt) {
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  call("stack_push", llvm_val[stack], tlctx->get_constant(stack->max_size),
       tlctx->get_constant(stack->element_size_in_bytes()));
  auto primal_ptr = call("stack_top_primal", llvm_val[stack],
                         tlctx->get_constant(stack->element_size_in_bytes()));
  primal_ptr = builder->CreateBitCast(
      primal_ptr,
      llvm::PointerType::get(tlctx->get_data_type(stmt->ret_type), 0));
  builder->CreateStore(llvm_val[stmt->v], primal_ptr);
}

void TaskCodeGenLLVM::visit(AdStackLoadTopStmt *stmt) {
  TI_ASSERT(stmt->return_ptr == false);
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  auto primal_ptr = call("stack_top_primal", llvm_val[stack],
                         tlctx->get_constant(stack->element_size_in_bytes()));
  auto primal_ty = tlctx->get_data_type(stmt->ret_type);
  primal_ptr =
      builder->CreateBitCast(primal_ptr, llvm::PointerType::get(primal_ty, 0));
  llvm_val[stmt] = builder->CreateLoad(primal_ty, primal_ptr);
}

void TaskCodeGenLLVM::visit(AdStackLoadTopAdjStmt *stmt) {
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  auto adjoint = call("stack_top_adjoint", llvm_val[stack],
                      tlctx->get_constant(stack->element_size_in_bytes()));
  auto adjoint_ty = tlctx->get_data_type(stmt->ret_type);
  adjoint =
      builder->CreateBitCast(adjoint, llvm::PointerType::get(adjoint_ty, 0));
  llvm_val[stmt] = builder->CreateLoad(adjoint_ty, adjoint);
}

void TaskCodeGenLLVM::visit(AdStackAccAdjointStmt *stmt) {
  auto stack = stmt->stack->as<AdStackAllocaStmt>();
  auto adjoint_ptr = call("stack_top_adjoint", llvm_val[stack],
                          tlctx->get_constant(stack->element_size_in_bytes()));
  auto adjoint_ty = tlctx->get_data_type(stack->ret_type);
  adjoint_ptr = builder->CreateBitCast(adjoint_ptr,
                                       llvm::PointerType::get(adjoint_ty, 0));
  auto old_val = builder->CreateLoad(adjoint_ty, adjoint_ptr);
  TI_ASSERT(is_real(stmt->v->ret_type));
  auto new_val = builder->CreateFAdd(old_val, llvm_val[stmt->v]);
  builder->CreateStore(new_val, adjoint_ptr);
}

void TaskCodeGenLLVM::visit(RangeAssumptionStmt *stmt) {
  llvm_val[stmt] = llvm_val[stmt->input];
}

void TaskCodeGenLLVM::visit(LoopUniqueStmt *stmt) {
  llvm_val[stmt] = llvm_val[stmt->input];
}

void TaskCodeGenLLVM::visit_call_bitcode(ExternalFuncCallStmt *stmt) {
  TI_ASSERT(stmt->type == ExternalFuncCallStmt::BITCODE);
  std::vector<llvm::Value *> arg_values;
  for (const auto &s : stmt->arg_stmts)
    arg_values.push_back(llvm_val[s]);
  // Link external module to the core module
  if (linked_modules.find(stmt->bc_filename) == linked_modules.end()) {
    linked_modules.insert(stmt->bc_filename);
    std::unique_ptr<llvm::Module> external_module =
        module_from_bitcode_file(stmt->bc_filename, llvm_context);
    auto *func_ptr = external_module->getFunction(stmt->bc_funcname);
    TI_ASSERT_INFO(func_ptr != nullptr, "{} is not found in {}.",
                   stmt->bc_funcname, stmt->bc_filename);
    auto link_error =
        llvm::Linker::linkModules(*module, std::move(external_module));
    TI_ASSERT(!link_error);
  }
  // Retrieve function again. Do it here to detect name conflicting.
  auto *func_ptr = module->getFunction(stmt->bc_funcname);
  // Convert pointer type from a[n * m] to a[n][m]
  for (int i = 0; i < func_ptr->getFunctionType()->getNumParams(); ++i) {
    TI_ASSERT_INFO(func_ptr->getArg(i)->getType()->getTypeID() ==
                       arg_values[i]->getType()->getTypeID(),
                   "TypeID {} != {} with {}",
                   (int)func_ptr->getArg(i)->getType()->getTypeID(),
                   (int)arg_values[i]->getType()->getTypeID(), i);
    auto tmp_value = arg_values[i];
    arg_values[i] =
        builder->CreatePointerCast(tmp_value, func_ptr->getArg(i)->getType());
  }
  call(func_ptr, arg_values);
}

void TaskCodeGenLLVM::visit_call_shared_object(ExternalFuncCallStmt *stmt) {
  TI_ASSERT(stmt->type == ExternalFuncCallStmt::SHARED_OBJECT);
  std::vector<llvm::Type *> arg_types;
  std::vector<llvm::Value *> arg_values;

  for (const auto &s : stmt->arg_stmts) {
    arg_types.push_back(tlctx->get_data_type(s->ret_type));
    arg_values.push_back(llvm_val[s]);
  }

  for (const auto &s : stmt->output_stmts) {
    auto t = tlctx->get_data_type(s->ret_type);
    auto ptr = llvm::PointerType::get(t, 0);
    arg_types.push_back(ptr);
    arg_values.push_back(llvm_val[s]);
  }

  auto func_type = llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                           arg_types, false);
  auto func_ptr_type = llvm::PointerType::get(func_type, 0);

  auto addr = tlctx->get_constant((std::size_t)stmt->so_func);
  auto func = builder->CreateIntToPtr(addr, func_ptr_type);
  call(func, func_type, arg_values);
}

void TaskCodeGenLLVM::visit(ExternalFuncCallStmt *stmt) {
  TI_NOT_IMPLEMENTED
}

void TaskCodeGenLLVM::visit(MeshPatchIndexStmt *stmt) {
  llvm_val[stmt] = get_arg(2);
}

void TaskCodeGenLLVM::visit(MatrixInitStmt *stmt) {
  auto type = tlctx->get_data_type(stmt->ret_type->as<TensorType>());
  llvm::Value *vec = llvm::UndefValue::get(type);
  for (int i = 0; i < stmt->values.size(); ++i) {
    auto *elem = llvm_val[stmt->values[i]];
    if (codegen_vector_type(compile_config)) {
      TI_ASSERT(llvm::dyn_cast<llvm::VectorType>(type));
      vec = builder->CreateInsertElement(vec, elem, i);
    } else {
      TI_ASSERT(llvm::dyn_cast<llvm::ArrayType>(type));
      vec = builder->CreateInsertValue(vec, elem, i);
    }
  }
  llvm_val[stmt] = vec;
}

void TaskCodeGenLLVM::eliminate_unused_functions() {
  TaichiLLVMContext::eliminate_unused_functions(
      module.get(), [&](std::string func_name) {
        for (auto &task : offloaded_tasks) {
          if (task.name == func_name)
            return true;
        }
        return false;
      });
}

FunctionCreationGuard TaskCodeGenLLVM::get_function_creation_guard(
    std::vector<llvm::Type *> argument_types,
    const std::string &func_name) {
  return FunctionCreationGuard(this, argument_types, func_name);
}

void TaskCodeGenLLVM::initialize_context() {
  TI_ASSERT(tlctx != nullptr);
  llvm_context = tlctx->get_this_thread_context();
  builder = std::make_unique<llvm::IRBuilder<>>(*llvm_context);
  if (compile_config.fast_math) {
    llvm::FastMathFlags fast_flags;
    fast_flags.setNoInfs();
    fast_flags.setNoSignedZeros();
    fast_flags.setAllowReassoc();
    builder->setFastMathFlags(fast_flags);
  }
}

llvm::Value *TaskCodeGenLLVM::get_arg(int i) {
  std::vector<llvm::Value *> args;
  for (auto &arg : func->args()) {
    args.push_back(&arg);
  }
  return args[i];
}

llvm::Value *TaskCodeGenLLVM::get_context() {
  return get_arg(0);
}

llvm::Value *TaskCodeGenLLVM::get_tls_base_ptr() {
  return get_arg(1);
}

llvm::Type *TaskCodeGenLLVM::get_tls_buffer_type() {
  return llvm::Type::getInt8PtrTy(*llvm_context);
}

std::vector<llvm::Type *> TaskCodeGenLLVM::get_xlogue_argument_types() {
  return {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
          get_tls_buffer_type()};
}

std::vector<llvm::Type *> TaskCodeGenLLVM::get_mesh_xlogue_argument_types() {
  return {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
          get_tls_buffer_type(), tlctx->get_data_type<uint32_t>()};
}

llvm::Type *TaskCodeGenLLVM::get_xlogue_function_type() {
  return llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                 get_xlogue_argument_types(), false);
}

llvm::Type *TaskCodeGenLLVM::get_mesh_xlogue_function_type() {
  return llvm::FunctionType::get(llvm::Type::getVoidTy(*llvm_context),
                                 get_mesh_xlogue_argument_types(), false);
}

llvm::PointerType *TaskCodeGenLLVM::get_integer_ptr_type(int bits) {
  switch (bits) {
    case 8:
      return llvm::Type::getInt8PtrTy(*llvm_context);
    case 16:
      return llvm::Type::getInt16PtrTy(*llvm_context);
    case 32:
      return llvm::Type::getInt32PtrTy(*llvm_context);
    case 64:
      return llvm::Type::getInt64PtrTy(*llvm_context);
    default:
      break;
  }
  TI_ERROR("No compatible " + std::to_string(bits) + " bits integer ptr type.");
  return nullptr;
}

llvm::IntegerType *TaskCodeGenLLVM::get_integer_type(int bits) {
  switch (bits) {
    case 8:
      return llvm::Type::getInt8Ty(*llvm_context);
    case 16:
      return llvm::Type::getInt16Ty(*llvm_context);
    case 32:
      return llvm::Type::getInt32Ty(*llvm_context);
    case 64:
      return llvm::Type::getInt64Ty(*llvm_context);
    default:
      break;
  }
  TI_ERROR("No compatible " + std::to_string(bits) + " bits integer type.");
  return nullptr;
}

llvm::Value *TaskCodeGenLLVM::get_root(int snode_tree_id) {
  return call("LLVMRuntime_get_roots", get_runtime(),
              tlctx->get_constant(snode_tree_id));
}

llvm::Value *TaskCodeGenLLVM::get_runtime() {
  auto runtime_ptr = call("RuntimeContext_get_runtime", get_context());
  return builder->CreateBitCast(
      runtime_ptr, llvm::PointerType::get(get_runtime_type("LLVMRuntime"), 0));
}

llvm::Value *TaskCodeGenLLVM::emit_struct_meta(SNode *snode) {
  auto obj = emit_struct_meta_object(snode);
  TI_ASSERT(obj != nullptr);
  return obj->ptr;
}

void TaskCodeGenLLVM::emit_to_module() {
  TI_AUTO_PROF
  ir->accept(this);
}

LLVMCompiledTask TaskCodeGenLLVM::run_compilation() {
  // Final lowering
  auto offload_to_executable = [](IRNode *ir, const CompileConfig &config,
                                  const Kernel *kernel) {
    bool verbose = config.print_ir;
    if (kernel->is_accessor && !config.print_accessor_ir) {
      verbose = false;
    }
    irpass::offload_to_executable(
        ir, config, kernel, verbose,
        /*determine_ad_stack_size=*/kernel->autodiff_mode ==
            AutodiffMode::kReverse,
        /*lower_global_access=*/true,
        /*make_thread_local=*/config.make_thread_local,
        /*make_block_local=*/
        is_extension_supported(config.arch, Extension::bls) &&
            config.make_block_local);
  };

  offload_to_executable(ir, compile_config, kernel);

  emit_to_module();
  eliminate_unused_functions();

  if (compile_config.arch == Arch::cuda) {
    // CUDA specific metadata
    for (const auto &task : offloaded_tasks) {
      llvm::Function *func = module->getFunction(task.name);
      TI_ASSERT(func);
      tlctx->mark_function_as_cuda_kernel(func, task.block_dim);
    }
  } else if (compile_config.arch == Arch::amdgpu) {
    for (const auto &task : offloaded_tasks) {
      llvm::Function *func = module->getFunction(task.name);
      TI_ASSERT(func);
      tlctx->mark_function_as_amdgpu_kernel(func);
    }
  }

  return {std::move(offloaded_tasks), std::move(module),
          std::move(used_tree_ids), std::move(struct_for_tls_sizes)};
}

llvm::Value *TaskCodeGenLLVM::create_xlogue(std::unique_ptr<Block> &block) {
  llvm::Value *xlogue;

  auto xlogue_type = get_xlogue_function_type();
  auto xlogue_ptr_type = llvm::PointerType::get(xlogue_type, 0);

  if (block) {
    auto guard = get_function_creation_guard(get_xlogue_argument_types());
    block->accept(this);
    xlogue = guard.body;
  } else {
    xlogue = llvm::ConstantPointerNull::get(xlogue_ptr_type);
  }

  return xlogue;
}

llvm::Value *TaskCodeGenLLVM::create_mesh_xlogue(
    std::unique_ptr<Block> &block) {
  llvm::Value *xlogue;

  auto xlogue_type = get_mesh_xlogue_function_type();
  auto xlogue_ptr_type = llvm::PointerType::get(xlogue_type, 0);

  if (block) {
    auto guard = get_function_creation_guard(get_mesh_xlogue_argument_types());
    block->accept(this);
    xlogue = guard.body;
  } else {
    xlogue = llvm::ConstantPointerNull::get(xlogue_ptr_type);
  }

  return xlogue;
}

void TaskCodeGenLLVM::visit(ReferenceStmt *stmt) {
  llvm_val[stmt] = llvm_val[stmt->var];
}

void TaskCodeGenLLVM::visit(FuncCallStmt *stmt) {
  if (!func_map.count(stmt->func)) {
    auto guard = get_function_creation_guard(
        {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0)},
        stmt->func->get_name());
    const Callable *old_callable = current_callable;
    current_callable = stmt->func;
    func_map.insert({stmt->func, guard.body});
    stmt->func->ir->accept(this);
    current_callable = old_callable;
  }
  llvm::Function *llvm_func = func_map[stmt->func];
  auto *new_ctx = create_entry_block_alloca(get_runtime_type("RuntimeContext"));
  call("RuntimeContext_set_runtime", new_ctx, get_runtime());
  if (!stmt->func->parameter_list.empty()) {
    auto *buffer =
        create_entry_block_alloca(tlctx->get_data_type(stmt->func->args_type));
    set_args_ptr(stmt->func, new_ctx, buffer);
    set_struct_to_buffer(stmt->func->args_type, buffer, stmt->args);
  }
  llvm::Value *result_buffer = nullptr;
  if (!stmt->func->rets.empty()) {
    auto *ret_type = tlctx->get_data_type(stmt->func->ret_type);
    result_buffer = create_entry_block_alloca(ret_type);
    auto *result_buffer_u64 = builder->CreatePointerCast(
        result_buffer,
        llvm::PointerType::get(tlctx->get_data_type<uint64>(), 0));
    call("RuntimeContext_set_result_buffer", new_ctx, result_buffer_u64);
  }
  call(llvm_func, new_ctx);
  llvm_val[stmt] = result_buffer;
}

void TaskCodeGenLLVM::visit(GetElementStmt *stmt) {
  auto *struct_type = tlctx->get_data_type(stmt->src->ret_type.ptr_removed());
  std::vector<llvm::Value *> index;
  index.reserve(stmt->index.size() + 1);
  index.push_back(tlctx->get_constant(0));
  for (auto &i : stmt->index) {
    index.push_back(tlctx->get_constant(i));
  }
  auto *gep = builder->CreateGEP(struct_type, llvm_val[stmt->src], index);
  auto *val = builder->CreateLoad(tlctx->get_data_type(stmt->ret_type), gep);
  llvm_val[stmt] = val;
}

void TaskCodeGenLLVM::set_struct_to_buffer(
    llvm::Value *buffer,
    llvm::Type *buffer_type,
    const std::vector<Stmt *> &elements,
    const Type *current_type,
    int &current_element,
    std::vector<llvm::Value *> &current_index) {
  if (auto primitive_type = current_type->cast<PrimitiveType>()) {
    TI_ASSERT((Type *)elements[current_element]->ret_type == current_type);
    auto *gep = builder->CreateGEP(buffer_type, buffer, current_index);
    builder->CreateStore(llvm_val[elements[current_element]], gep);
    current_element++;
  } else if (auto pointer_type = current_type->cast<PointerType>()) {
    TI_ASSERT((Type *)elements[current_element]->ret_type == current_type);
    auto *gep = builder->CreateGEP(buffer_type, buffer, current_index);
    builder->CreateStore(llvm_val[elements[current_element]], gep);
    current_element++;
  } else if (auto struct_type = current_type->cast<StructType>()) {
    int i = 0;
    for (const auto &element : struct_type->elements()) {
      current_index.push_back(tlctx->get_constant(i++));
      set_struct_to_buffer(buffer, buffer_type, elements, element.type,
                           current_element, current_index);
      current_index.pop_back();
    }
  } else if (auto tensor_type = current_type->cast<TensorType>()) {
    int num_elements = tensor_type->get_num_elements();
    Type *element_type = tensor_type->get_element_type();
    for (int i = 0; i < num_elements; i++) {
      current_index.push_back(tlctx->get_constant(i));
      set_struct_to_buffer(buffer, buffer_type, elements, element_type,
                           current_element, current_index);
      current_index.pop_back();
    }
  } else {
    TI_INFO("{}", current_type->to_string());
    TI_NOT_IMPLEMENTED
  }
}

void TaskCodeGenLLVM::set_struct_to_buffer(
    const StructType *struct_type,
    llvm::Value *buffer,
    const std::vector<Stmt *> &elements) {
  auto buffer_type = tlctx->get_data_type(struct_type);
  buffer = builder->CreatePointerCast(buffer,
                                      llvm::PointerType::get(buffer_type, 0));
  int current_element = 0;
  std::vector<llvm::Value *> current_index = {tlctx->get_constant(0)};
  set_struct_to_buffer(buffer, buffer_type, elements, struct_type,
                       current_element, current_index);
}

llvm::Value *TaskCodeGenLLVM::get_argpack_arg(const std::vector<int> &arg_id,
                                              int arg_depth,
                                              bool create_load) {
  const std::vector<int> indices_l(arg_id.begin(), arg_id.begin() + arg_depth);
  const std::vector<int> indices_r(arg_id.begin() + arg_depth, arg_id.end());
  llvm::Value *data_ptr_value;
  auto argpack_iterator =
      std::find_if(current_callable->argpack_types.begin(),
                   current_callable->argpack_types.end(),
                   [&](const auto &kv) { return kv.first == indices_l; });
  TI_ASSERT(argpack_iterator != current_callable->argpack_types.end());
  const auto *argpack_type = (*argpack_iterator).second;
  auto *arg_type = argpack_type->get_element_type(indices_r);
  if (arg_depth > 1) {
    auto key = arg_id;
    key.back() = TypeFactory::DATA_PTR_POS_IN_ARGPACK;
    data_ptr_value = get_argpack_arg(key, arg_depth - 1, true);
  } else {
    auto indices_data_ptr = indices_l;
    indices_data_ptr.push_back(TypeFactory::DATA_PTR_POS_IN_ARGPACK);
    data_ptr_value = get_struct_arg(indices_data_ptr, true);
  }
  std::vector<llvm::Value *> gep_index;
  gep_index.reserve(indices_r.size());
  gep_index.push_back(tlctx->get_constant(0));
  for (int ind : indices_r) {
    gep_index.push_back(tlctx->get_constant(ind));
  }
  auto *gep = builder->CreateGEP(tlctx->get_data_type(argpack_type),
                                 data_ptr_value, gep_index);
  if (!create_load) {
    return gep;
  }
  return builder->CreateLoad(tlctx->get_data_type(arg_type), gep);
}

llvm::Value *TaskCodeGenLLVM::get_struct_arg(const std::vector<int> &index,
                                             bool create_load) {
  auto *args_ptr = get_args_ptr(current_callable, get_context());
  auto *args_type = current_callable->args_type;
  auto *arg_type = args_type->get_element_type(index);
  std::vector<llvm::Value *> gep_index;
  gep_index.reserve(index.size() + 1);
  gep_index.push_back(tlctx->get_constant(0));
  for (int ind : index) {
    gep_index.push_back(tlctx->get_constant(ind));
  }
  auto *gep =
      builder->CreateGEP(tlctx->get_data_type(args_type), args_ptr, gep_index);
  if (!create_load) {
    return gep;
  }
  return builder->CreateLoad(tlctx->get_data_type(arg_type), gep);
}

llvm::Value *TaskCodeGenLLVM::get_args_ptr(const Callable *callable,
                                           llvm::Value *context) {
  auto *runtime_context_type = get_runtime_type("RuntimeContext");
  auto *args_type = tlctx->get_data_type(callable->args_type);
  auto *zero = tlctx->get_constant(0);
  // The address of the arg buffer is the first element of RuntimeContext
  auto *args_ptr =
      builder->CreateGEP(runtime_context_type, context, {zero, zero});
  // casting from i8 ** to args_type **
  args_ptr = builder->CreatePointerCast(
      args_ptr,
      llvm::PointerType::get(llvm::PointerType::get(args_type, 0), 0));
  // loading the address of the arg buffer (args_type *)
  args_ptr =
      builder->CreateLoad(llvm::PointerType::get(args_type, 0), args_ptr);
  return args_ptr;
}
void TaskCodeGenLLVM::set_args_ptr(Callable *callable,
                                   llvm::Value *context,
                                   llvm::Value *ptr) {
  auto *runtime_context_type = get_runtime_type("RuntimeContext");
  auto *args_type = tlctx->get_data_type(callable->args_type);
  auto *zero = tlctx->get_constant(0);
  // The address of the arg buffer is the first element of RuntimeContext
  auto *args_ptr =
      builder->CreateGEP(runtime_context_type, context, {zero, zero});
  // casting from i8 ** to args_type **
  args_ptr = builder->CreatePointerCast(
      args_ptr,
      llvm::PointerType::get(llvm::PointerType::get(args_type, 0), 0));
  // storing the address of the arg buffer (args_type *)
  builder->CreateStore(ptr, args_ptr);
};

LLVMCompiledTask LLVMCompiledTask::clone() const {
  return {tasks, llvm::CloneModule(*module), used_tree_ids,
          struct_for_tls_sizes};
}

LLVMCompiledKernel LLVMCompiledKernel::clone() const {
  return {tasks, llvm::CloneModule(*module)};
}

}  // namespace taichi::lang

#endif  // #ifdef TI_WITH_LLVM
