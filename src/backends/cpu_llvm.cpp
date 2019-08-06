#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../util.h"
#include "cpu.h"
#include "../program.h"
#include "../ir.h"

#if defined(TLANG_WITH_LLVM)
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm_jit.h"

#endif

TLANG_NAMESPACE_BEGIN

#if defined(TLANG_WITH_LLVM)

using namespace llvm;

class CPULLVMCodeGen : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  TaichiLLVMContext *tlctx;
  llvm::LLVMContext *llvm_context;
  TaichiLLVMJIT *jit;
  llvm::IRBuilder<> builder;
  std::unique_ptr<Module> module;

  CodeGenBase *codegen;
  Program::Kernel *kernel;
  llvm::Function *func;
  llvm::Constant *func_printf;

  CPULLVMCodeGen(CodeGenBase *codegen)
      : tlctx(&get_current_program().llvm_context),
        llvm_context(tlctx->ctx.get()),
        jit(tlctx->jit.get()),
        builder(*llvm_context) {
    using namespace llvm;
    module = tlctx->clone_struct_module();
    for (auto &f : *module) {
      f.setLinkage(Function::PrivateLinkage);
    }
    current_struct_for = nullptr;

    std::vector<Type *> args{
        llvm::PointerType::get(Type::getInt8Ty(*llvm_context), 0)};
    auto *FT =
        llvm::FunctionType::get(Type::getInt32Ty(*llvm_context), args, false);

    func =
        Function::Create(FT, Function::ExternalLinkage, "kernel", module.get());

    module->setDataLayout(jit->getDataLayout());

    auto func_printf_type = llvm::FunctionType::get(
        IntegerType::getInt32Ty(*llvm_context),
        PointerType::get(Type::getInt8Ty(*llvm_context), 0), true);
    func_printf = Function::Create(func_printf_type, Function::ExternalLinkage,
                                   "printf", module.get());
  }

  void gen(IRNode *node) {
    BasicBlock *bb = BasicBlock::Create(*llvm_context, "entry", func);
    builder.SetInsertPoint(bb);
    node->accept(this);
    builder.CreateRet(const_int32(0));

    TC_INFO("Module IR");
    module->print(errs(), nullptr);
    TC_ASSERT(!llvm::verifyFunction(*func, &errs()));

    llvm::cantFail(jit->addModule(std::move(module)));

    auto kernel_symbol = llvm::cantFail(jit->lookup("kernel"));
    TC_ASSERT_INFO(kernel_symbol, "Function not found");

    auto f = (int32(*)(void *))(void *)(kernel_symbol.getAddress());
    auto context = get_current_program().get_context();
    f(&context);

    TC_INFO("Exiting...");

    exit(-1);
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    TC_NOT_IMPLEMENTED
    codegen->emit(f, std::forward<Args>(args)...);
  }

  static void run(CodeGenBase *codegen, IRNode *node, Program::Kernel *kernel) {
    auto p = CPULLVMCodeGen(codegen);
    p.kernel = kernel;
    p.gen(node);
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    if (stmt->ret_type.data_type == DataType::i32) {
      stmt->value = builder.CreateAlloca(llvm::Type::getInt32Ty(*llvm_context),
                                         (unsigned)0);
    } else {
      TC_NOT_IMPLEMENTED
    }
  }

  void visit(RandStmt *stmt) {
    TC_ASSERT(stmt->ret_type.data_type == DataType::f32);
    emit("const auto {} = {}::rand();", stmt->raw_name(),
         stmt->ret_data_type_name());
  }

  void visit(UnaryOpStmt *stmt) {
    if (stmt->op_type != UnaryOpType::cast) {
      emit("const {} {}({}({}));", stmt->ret_data_type_name(), stmt->raw_name(),
           unary_op_type_name(stmt->op_type), stmt->operand->raw_name());
    } else {
      if (stmt->cast_by_value) {
        llvm::CastInst::CastOps cast_op;
        auto from = stmt->operand->ret_type.data_type;
        auto to = stmt->cast_type;
        if (from == DataType::f32 && to == DataType::i32) {
          cast_op = llvm::Instruction::CastOps::FPToSI;
        } else if (from == DataType::i32 && to == DataType::f32) {
          cast_op = llvm::Instruction::CastOps::SIToFP;
        } else {
          cast_op = llvm::Instruction::CastOps::FPToSI;
          TC_NOT_IMPLEMENTED;
        }
        stmt->value = builder.CreateCast(cast_op, stmt->operand->value,
                                         tlctx->get_data_type(stmt->cast_type));
      } else {
        TC_ASSERT(data_type_size(stmt->ret_type.data_type) ==
                  data_type_size(stmt->cast_type));
        stmt->value = builder.CreateBitCast(
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
      TC_NOT_IMPLEMENTED;
    }
    return nullptr;
  }

  void visit(BinaryOpStmt *stmt) {
    auto op = stmt->op_type;
    if (op == BinaryOpType::add) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder.CreateFAdd(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder.CreateAdd(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::sub) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder.CreateFSub(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder.CreateSub(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::mul) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder.CreateFMul(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder.CreateMul(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::div) {
      if (is_real(stmt->ret_type.data_type)) {
        stmt->value = builder.CreateFDiv(stmt->lhs->value, stmt->rhs->value);
      } else {
        stmt->value = builder.CreateSDiv(stmt->lhs->value, stmt->rhs->value);
      }
    } else if (op == BinaryOpType::mod) {
      stmt->value = builder.CreateSRem(stmt->lhs->value, stmt->rhs->value);
    } else if (is_comparison(op)) {
      if (op == BinaryOpType::cmp_eq) {
        if (is_real(stmt->lhs->ret_type.data_type)) {
          stmt->value = builder.CreateSExt(
              builder.CreateFCmpOEQ(stmt->lhs->value, stmt->rhs->value),
              llvm_type(DataType::i32));
        } else {
          stmt->value = builder.CreateSExt(
              builder.CreateICmpEQ(stmt->lhs->value, stmt->rhs->value),
              llvm_type(DataType::i32));
        }
      } else {
        TC_NOT_IMPLEMENTED
      }
    } else {
      TC_NOT_IMPLEMENTED
    }
  }

  void visit(TernaryOpStmt *stmt) {
    TC_ASSERT(stmt->op_type == TernaryOpType::select);
    stmt->value = builder.CreateSelect(
        builder.CreateTrunc(stmt->op1->value, llvm_type(DataType::i1)),
        stmt->op2->value, stmt->op3->value);
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements) {
      emit("if (any({})) {{", if_stmt->true_mask->raw_name());
      if_stmt->true_statements->accept(this);
      emit("}}");
    }
    if (if_stmt->false_statements) {
      emit("if (any({})) {{", if_stmt->false_mask->raw_name());
      if_stmt->false_statements->accept(this);
      emit("}}");
    }
  }

  void visit(PrintStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    std::vector<Value *> args;
    std::string format;
    if (stmt->stmt->ret_type.data_type == DataType::i32) {
      format = "%d";
    } else if (stmt->stmt->ret_type.data_type == DataType::f32) {
      format = "%f";
    } else {
      TC_NOT_IMPLEMENTED
    }
    args.push_back(builder.CreateGlobalStringPtr(
        ("[debug] " + stmt->str + " = " + format + "\n").c_str(),
        "format string"));
    args.push_back(stmt->stmt->value);

    stmt->value = builder.CreateCall(func_printf, args, "printf");
  }

  void visit(ConstStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    auto val = stmt->val[0];
    if (val.dt == DataType::f32) {
      stmt->value = llvm::ConstantFP::get(*llvm_context,
                                          llvm::APFloat(val.val_float32()));
    } else if (val.dt == DataType::i32) {
      stmt->value = llvm::ConstantInt::get(
          *llvm_context, llvm::APInt(32, val.val_int32(), true));
    } else {
      TC_NOT_IMPLEMENTED;
    }
  }

  void visit(WhileControlStmt *stmt) {
    emit("{} = bit_and({}, {});", stmt->mask->raw_name(),
         stmt->mask->raw_name(), stmt->cond->raw_name());
    emit("if (!any({})) break;", stmt->mask->raw_name());
  }

  void visit(WhileStmt *stmt) {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(StructForStmt *for_stmt) {
  }

  AllocaInst *CreateEntryBlockAlloca(llvm::Type *type,
                                     const std::string &name) {
    llvm::IRBuilder<> TmpB(&func->getEntryBlock(),
                           func->getEntryBlock().begin());
    return TmpB.CreateAlloca(type, 0, name.c_str());
  }

  void increment(llvm::Value *ptr, llvm::Value *value) {
    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(ptr), value), ptr);
  }

  llvm::Value *const_int32(int32 val) {
    return llvm::ConstantInt::get(*llvm_context, llvm::APInt(32, val, true));
  }

  void visit(RangeForStmt *for_stmt) {
    // auto entry = builder.GetInsertBlock();

    BasicBlock *body = BasicBlock::Create(*llvm_context, "loop_body", func);
    BasicBlock *after_loop = BasicBlock::Create(*llvm_context, "block", func);
    builder.CreateStore(const_int32(0), for_stmt->loop_var->value);
    builder.CreateBr(body);

    // body cfg
    builder.SetInsertPoint(body);

    /*
    auto phi = builder.CreatePHI(llvm::Type::getInt32Ty(llvm_context), 2);
    auto loop_inc = builder.CreateAdd(
        phi, llvm::ConstantInt::get(llvm_context, llvm::APInt(32, 1, true)));
    phi->addIncoming(for_stmt->begin->value, entry);
    phi->addIncoming(loop_inc, body);
    */

    for_stmt->body->accept(this);

    increment(for_stmt->loop_var->value, const_int32(1));

    auto cond = builder.CreateICmp(
        llvm::CmpInst::Predicate::ICMP_SLT,
        builder.CreateLoad(for_stmt->loop_var->value), for_stmt->end->value);

    builder.CreateCondBr(cond, body, after_loop);

    // next cfg
    builder.SetInsertPoint(after_loop);
  }

  void visit(ArgLoadStmt *stmt) {
    emit("const {} {}({{context.get_arg<{}>({})}});",
         stmt->ret_data_type_name(), stmt->raw_name(),
         data_type_name(stmt->ret_type.data_type), stmt->arg_id);
  }

  void visit(LocalLoadStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    stmt->value = builder.CreateLoad(stmt->ptr[0].var->value);
  }

  void visit(LocalStoreStmt *stmt) {
    auto mask = stmt->parent->mask();
    if (mask) {
      TC_NOT_IMPLEMENTED
    } else {
      builder.CreateStore(stmt->data->value, stmt->ptr->value);
    }
  }

  void visit(SNodeOpStmt *stmt) {
    stmt->ret_type.data_type = DataType::i32;
    if (stmt->op_type == SNodeOpType::probe) {
      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
    }

    for (auto l = 0; l < stmt->width(); l++) {
      auto snode = stmt->snodes[l];
      auto indices = indices_str(snode, l, stmt->indices);

      emit("{{");
      if (stmt->op_type != SNodeOpType::activate &&
          stmt->op_type != SNodeOpType::probe) {
        emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
             snode->node_type_name, snode->node_type_name,
             make_list(indices, ""));
      }
      if (stmt->op_type == SNodeOpType::append) {
        TC_ASSERT(stmt->val->width() == 1);
        emit("{}_tmp->append({}({}[{}]));", snode->node_type_name,
             snode->ch[0]->node_type_name, stmt->val->raw_name(), l);
      } else if (stmt->op_type == SNodeOpType::clear) {
        emit("{}_tmp->clear();", snode->node_type_name);
      } else if (stmt->op_type == SNodeOpType::probe) {
        emit("{}[{}] = query_{}(root, {});", stmt->raw_name(), l,
             snode->node_type_name, make_list(indices, ""));
        if (snode->type == SNodeType::dynamic) {
          emit("if ({}[{}]) {{", stmt->raw_name(), l);
          emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
               snode->node_type_name, snode->node_type_name,
               make_list(indices, ""));
          emit("{}[{}] = {}_tmp->get_n();", stmt->raw_name(), l,
               snode->node_type_name);
          emit("}}");
        }
      } else if (stmt->op_type == SNodeOpType::activate) {
        emit("activate_{}(root, {});", snode->node_type_name,
             make_list(indices, ""));
      } else {
        TC_NOT_IMPLEMENTED
      }
      emit("}}");
    }
  }

  void visit(AtomicOpStmt *stmt) {
    auto mask = stmt->parent->mask();
    for (int l = 0; l < stmt->width(); l++) {
      if (mask) {
        emit("if ({}[{}]) ", mask->raw_name(), l);
      } else {
        TC_ASSERT(stmt->val->ret_type.data_type == DataType::f32 ||
                  stmt->val->ret_type.data_type == DataType::i32);
        TC_ASSERT(stmt->op_type == AtomicOpType::add);
        emit("atomic_add({}[{}], {}[{}]);", stmt->dest->raw_name(), l,
             stmt->val->raw_name(), l);
      }
    }
  }

  void visit(GlobalPtrStmt *stmt) {
    emit("{} *{}[{}];", data_type_name(stmt->ret_type.data_type),
         stmt->raw_name(), stmt->ret_type.width);
    for (int l = 0; l < stmt->ret_type.width; l++) {
      // Try to weaken here...
      std::vector<int> offsets(stmt->indices.size());

      auto snode = stmt->snodes[l];
      std::vector<std::string> indices(max_num_indices, "0");  // = "(root, ";
      for (int i = 0; i < (int)stmt->indices.size(); i++) {
        if (snode->physical_index_position[i] != -1) {
          // TC_ASSERT(snode->physical_index_position[i] != -1);
          indices[snode->physical_index_position[i]] =
              stmt->indices[i]->raw_name() + fmt::format("[{}]", l);
        }
      }
      std::string full_access = fmt::format(
          "{}[{}] = &{}_{}{}->val;", stmt->raw_name(), l,
          stmt->accessor_func_name(), stmt->snodes[l]->node_type_name,
          "(root, " + make_list(indices, "") + ")");

      bool weakened = false;
      if (current_struct_for &&
          snode->parent == current_struct_for->snode->parent) {
        bool identical_indices = false;
        bool all_offsets_zero = true;
        for (int i = 0; i < (int)stmt->indices.size(); i++) {
          auto ret = analysis::value_diff(stmt->indices[i], l,
                                          current_struct_for->loop_vars[i]);
          if (!ret.linear_related() || !ret.certain()) {
            identical_indices = false;
          }
          offsets[i] = ret.low;
          if (ret.low != 0)
            all_offsets_zero = false;
        }
        if (identical_indices) {
          // TC_WARN("Weakened addressing");
          weakened = true;

          std::string cond;
          cond = "true";
          // add safe guards...
          for (int i = 0; i < (int)stmt->indices.size(); i++) {
            if (offsets[i] == 0)
              continue;
            // TODO: fix hacky hardcoded name, make sure index same order as
            // snode indices
            std::string local_var = fmt::format(
                "index_{}_{}_local", snode->parent->node_type_name, i);
            int upper_bound = 1 << snode->parent->extractors[i].num_bits;
            if (offsets[i] == -1) {
              cond += fmt::format("&& {} > 0", local_var);
            } else if (offsets[i] >= 1) {
              cond += fmt::format("&& {} < {} - {}", local_var, upper_bound,
                                  offsets[i]);
            }
          }

          int offset = 0;
          int current_num_bits = 0;
          for (int i = (int)stmt->indices.size() - 1; i >= 0; i--) {
            offset += offsets[i] * (1 << current_num_bits);
            current_num_bits += snode->parent->extractors[i].num_bits;
          }

          emit("if ({}) {{", cond);
          emit("{}[{}] = &access_{}({}_cache, {}_loop + {})->val;",
               stmt->raw_name(), l, snode->node_type_name,
               snode->parent->node_type_name, snode->parent->node_type_name,
               offset);
          emit("}} else {{");
          emit("{}", full_access);
          emit("}}");
        }
      }
      if (!weakened) {
        emit("{}", full_access);
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    if (!current_program->config.force_vectorized_global_store) {
      for (int i = 0; i < stmt->data->ret_type.width; i++) {
        if (stmt->parent->mask()) {
          TC_ASSERT(stmt->width() == 1);
          emit("if ({}[{}])", stmt->parent->mask()->raw_name(), i);
        }
        emit("*({} *){}[{}] = {}[{}];",
             data_type_name(stmt->data->ret_type.data_type),
             stmt->ptr->raw_name(), i, stmt->data->raw_name(), i);
      }
    } else {
      emit("{}.store({}[0]);", stmt->data->raw_name(), stmt->ptr->raw_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    int width = stmt->width();
    if (get_current_program().config.attempt_vectorized_load_cpu &&
        width >= 4 && stmt->ptr->is<ElementShuffleStmt>()) {
      TC_ASSERT(stmt->ret_type.data_type == DataType::i32 ||
                stmt->ret_type.data_type == DataType::f32);
      bool loaded[width];
      for (int i = 0; i < width; i++)
        loaded[i] = false;

      auto shuffle = stmt->ptr->as<ElementShuffleStmt>();
      Stmt *statements[width];
      int offsets[width];

      for (int i = 0; i < width; i++) {
        auto src = shuffle->elements[i].stmt;
        if (shuffle->elements[i].stmt->is<IntegerOffsetStmt>()) {
          auto indir = src->as<IntegerOffsetStmt>();
          statements[i] = indir->input;
          offsets[i] = indir->offset;
        } else {
          statements[i] = src;
          offsets[i] = 0;
        }
      }

      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
      for (int i = 0; i < width; i++) {
        if (loaded[i])
          continue;
        bool mask[width];
        std::memset(mask, 0, sizeof(mask));
        mask[i] = true;
        for (int j = i + 1; j < width; j++) {
          if (statements[i] == statements[j]) {
            if ((j - i) * (int)sizeof(int32) == offsets[j] - offsets[i]) {
              mask[j] = true;
            }
          }
        }
        int imm_mask = 0;
        for (int j = width - 1; j >= 0; j--) {
          if (mask[j]) {
            loaded[j] = true;
          }
          imm_mask *= 2;
          imm_mask += (int)mask[j];
        }
        // load and blend in
        if (i == 0) {
          emit("{} = {}::load({}[0]);", stmt->raw_name(),
               stmt->ret_data_type_name(),
               shuffle->elements[i].stmt->raw_name());
        } else {
          emit("{} = blend<{}>({}, {}::load({}[0] - {}));", stmt->raw_name(),
               imm_mask, stmt->raw_name(), stmt->ret_data_type_name(),
               shuffle->elements[i].stmt->raw_name(), i);
        }
      }
    } else {
      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
      for (int i = 0; i < stmt->ret_type.width; i++) {
        emit("{}[{}] = *{}[{}];", stmt->raw_name(), i, stmt->ptr->raw_name(),
             i);
      }
    }
  }

  void visit(ElementShuffleStmt *stmt) {
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

  void visit(AssertStmt *stmt) {
    emit("#if defined(TL_DEBUG)");
    emit(R"(TC_ASSERT_INFO({}, "{}");)", stmt->val->raw_name(), stmt->text);
    emit("#endif");
  }

  void visit(OffsetAndExtractBitsStmt *stmt) {
    /*
    emit(R"(auto {} = ((({} + {}) >> {}) & ((1 << {}) - 1));)",
         stmt->raw_name(), stmt->offset, stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
    */

    auto shifted =
        builder.CreateAdd(stmt->value, tlctx->get_constant(stmt->offset));
    int mask = (1u << (stmt->bit_end - stmt->bit_begin)) - 1;
    stmt->value =
        builder.CreateAnd(builder.CreateLShr(shifted, stmt->bit_begin),
                          tlctx->get_constant(mask));
  }

  void visit(LinearizeStmt *stmt) {
    llvm::Value *val = tlctx->get_constant(0);
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      // val = fmt::format("({}) * {} + {}", val, stmt->strides[i],
      //                  stmt->inputs[i]->raw_name());

      val = builder.CreateAdd(
          builder.CreateMul(stmt->value, tlctx->get_constant(stmt->strides[i])),
          stmt->inputs[i]->value);
    }
    stmt->value = val;
  }

  void visit(IntegerOffsetStmt *stmt) {
    TC_NOT_IMPLEMENTED
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

  void visit(SNodeLookupStmt *stmt) {
    TC_NOT_IMPLEMENTED
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      parent = "root";
    }
    std::vector<std::string> global_indices(max_num_indices, "0");
    auto snode = stmt->snode;
    for (int i = 0; i < (int)stmt->global_indices.size(); i++) {
      if (snode->physical_index_position[i] != -1) {
        global_indices[snode->physical_index_position[i]] =
            stmt->global_indices[i]->raw_name() + fmt::format("[{}]", 0);
      }
    }
    if (stmt->activate && stmt->snode->type != SNodeType::place) {
      emit(R"({}->activate({}, {});)", parent, stmt->input_index->raw_name(),
           make_list(global_indices, "{"));
    }
    emit("auto {}_guarded = {}->look_up({});", stmt->raw_name(), parent,
         stmt->input_index->raw_name());
    if (!stmt->activate && snode->has_null()) {
      // safe guard with ambient node
      emit("if({}_guarded == nullptr) {}_guarded = &{}_ambient;",
           stmt->raw_name(), stmt->raw_name(), snode->node_type_name);
    }
    emit(R"(auto {} = {}_guarded;)", stmt->raw_name(), stmt->raw_name());
  }

  void visit(GetChStmt *stmt) {
    // emit("{} *{};", stmt->output_snode->data_type_name(),
    //     stmt->raw_name());
    if (stmt->output_snode->type == SNodeType::place) {
      emit(R"({} *{}[1] {{&{}->get{}()->val}};)",
           stmt->output_snode->data_type_name(), stmt->raw_name(),
           stmt->input_ptr->raw_name(), stmt->chid);
    } else {
      emit(R"(auto {} = {}->get{}();)", stmt->raw_name(),
           stmt->input_ptr->raw_name(), stmt->chid);
    }
  }
};

FunctionType CPUCodeGen::codegen_llvm() {
  CPULLVMCodeGen::run(this, kernel->ir, kernel);
  return nullptr;
}
#else

FunctionType CPUCodeGen::codegen_llvm() {
  TC_ERROR("LLVM not found");
}

#endif

TLANG_NAMESPACE_END
