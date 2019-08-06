#include "llvm/IR/Verifier.h"
#include <llvm/IR/IRBuilder.h>
#include "../ir.h"
#include "../program.h"
#include "struct_llvm.h"

TLANG_NAMESPACE_BEGIN

StructCompilerLLVM::StructCompilerLLVM() : StructCompiler() {
  creator = [] {
    TC_WARN("Data structure creation not implemented");
    return nullptr;
  };
  Program *prog = &get_current_program();
  tlctx = &prog->llvm_context;
  llvm_ctx = tlctx->ctx.get();
  module = llvm::make_unique<Module>("taichi struct", *llvm_ctx);
}

void StructCompilerLLVM::codegen(SNode &snode) {
  auto type = snode.type;
  llvm::Type *llvm_type = nullptr;

  Program *prog = &get_current_program();
  auto ctx = prog->llvm_context.ctx.get();

  // create children type that supports forking...

  std::vector<llvm::Type *> ch_types;
  for (int i = 0; i < snode.ch.size(); i++) {
    auto ch = llvm_types[snode.ch[i].get()];
    ch_types.push_back(ch);
  }

  auto ch_type =
      llvm::StructType::create(*ctx, ch_types, snode.node_type_name + "_ch");
  ch_type->setName(snode.node_type_name + "_ch");

  if (type == SNodeType::dense) {
    TC_ASSERT(snode._bitmasked == false);
    TC_ASSERT(snode._morton == false);
    llvm_type = llvm::ArrayType::get(ch_type, 1 << snode.total_num_bits);
  } else if (type == SNodeType::root) {
    llvm_type = ch_type;
  } else if (type == SNodeType::place) {
    if (snode.dt == DataType::f32) {
      llvm_type = llvm::Type::getFloatTy(*ctx);
    } else if (snode.dt == DataType::i32) {
      llvm_type = llvm::Type::getInt32Ty(*ctx);
    } else {
      TC_NOT_IMPLEMENTED
    }
  } else {
    TC_P(snode.type_name());
    TC_NOT_IMPLEMENTED;
  }

  if (snode.has_null()) {
    if (get_current_program().config.arch == Arch::gpu) {
      emit("__device__ __constant__ {}::child_type *{}_ambient_ptr;",
           snode.node_type_name, snode.node_type_name);
    }
    emit("{}::child_type {}_ambient;", snode.node_type_name,
         snode.node_type_name);
  } else if (snode.type == SNodeType::place) {
    emit("{} {}_ambient;", snode.node_type_name, snode.node_type_name);
  }

  TC_ASSERT(llvm_type != nullptr);
  llvm_types[&snode] = llvm_type;
}

void StructCompilerLLVM::generate_leaf_accessors(SNode &snode) {
  auto type = snode.type;
  stack.push_back(&snode);

  bool is_leaf = type == SNodeType::place;

  auto llvm_index_type = llvm::Type::getInt32Ty(*llvm_ctx);

  if (!is_leaf) {
    // Chain accessors for non-leaf nodes
    TC_ASSERT(snode.ch.size() > 0);
    TC_ASSERT_INFO(
        snode.type == SNodeType::dense || snode.type == SNodeType::root,
        "TODO: deal with child nullptrs when sparse");
    for (int i = 0; i < (int)snode.ch.size(); i++) {
      auto ch = snode.ch[i];
      llvm::Type *parent_type = llvm_types[&snode];
      llvm::Type *child_type = llvm_types[ch.get()];
      llvm::Type *parent_ptr_type = llvm::PointerType::get(parent_type, 0);
      llvm::Type *child_ptr_type = llvm::PointerType::get(child_type, 0);

      auto ft = llvm::FunctionType::get(
          child_ptr_type, {parent_ptr_type, llvm_index_type}, false);
      auto accessor = llvm::Function::Create(
          ft, llvm::Function::InternalLinkage,
          "chain_accessor_" + ch->get_name(), module.get());
      accessor->addAttribute(
          0, llvm::Attribute::get(*llvm_ctx, llvm::Attribute::AlwaysInline));
      auto bb = BasicBlock::Create(*llvm_ctx, "body", accessor);
      llvm::IRBuilder<> builder(bb, bb->begin());
      std::vector<Value *> args;
      for (auto &arg : accessor->args()) {
        args.push_back(&arg);
      }
      llvm::Value *parent_ptr = args[0];

      args[0]->setName("parent_ptr");
      args[1]->setName("index");

      llvm::Value *index = args[1];
      llvm::Value *fork = nullptr;

      if (snode.type == SNodeType::dense) {
        fork = builder.CreateGEP(
            parent_ptr,
            {llvm::ConstantInt::get(llvm::Type::getInt32Ty(*llvm_ctx), 0),
             index});
      } else if (snode.type == SNodeType::root) {
        fork = parent_ptr;
      }
      auto ret = builder.CreateStructGEP(fork, i);
      builder.CreateRet(ret);

      TC_WARN_IF(llvm::verifyFunction(*accessor, &errs()),
                 "function verification failed");

      chain_accessors[ch.get()] = accessor;
    }
    emit("");
  }
  // SNode::place & indirect
  // emit end2end accessors for leaf (place) nodes, using chain accessors
  TC_ASSERT(max_num_indices == 4);
  constexpr int mode_weak_access = 0;
  constexpr int mode_strong_access = 1;
  constexpr int mode_activate = 2;
  constexpr int mode_query = 3;

  std::vector<std::string> verbs(4);
  verbs[mode_weak_access] = "weak_access";
  verbs[mode_strong_access] = "access";
  verbs[mode_activate] = "activate";
  verbs[mode_query] = "query";

  TC_WARN("TODO: mode_activate, mode_query");

  for (auto mode : {mode_weak_access, mode_strong_access}) {
    if (mode == mode_weak_access || !is_leaf)
      continue;
    bool is_access = mode == mode_weak_access || mode == mode_strong_access;
    auto verb = verbs[mode];
    auto ret_type =
        mode == mode_query ? "bool" : fmt::format("{} *", snode.node_type_name);
    emit(
        "TLANG_ACCESSOR TC_EXPORT {} {}_{}(void *root, int i0=0, int i1=0, "
        "int "
        "i2=0, "
        "int i3=0) {{",
        ret_type, verb, snode.node_type_name);
    std::vector<llvm::Type *> arg_types{
        llvm::PointerType::get(llvm_types[&root], 0)};
    for (int i = 0; i < max_num_indices; i++) {
      arg_types.push_back(llvm_index_type);
    }
    auto ft = llvm::FunctionType::get(
        llvm::PointerType::get(tlctx->get_data_type(snode.dt), 0), arg_types,
        false);
    auto accessor = llvm::Function::Create(
        ft, llvm::Function::ExternalLinkage,
        "leaf_accessor_" + snode.node_type_name, module.get());
    auto bb = BasicBlock::Create(*llvm_ctx, "body", accessor);
    llvm::IRBuilder<> builder(bb, bb->begin());
    std::vector<Value *> args;
    for (auto &arg : accessor->args()) {
      args.push_back(&arg);
    }
    args[0]->setName("root_ptr");
    for (int i = 0; i < max_num_indices; i++) {
      args[1 + i]->setName(fmt::format("index{}", i));
    }
    emit("int tmp;");
    emit("auto n0 = ({} *)root;", root_type);

    llvm::Value *node = args[0];
    for (int i = 0; i + 1 < (int)stack.size(); i++) {
      emit("tmp = 0;", i);
      llvm::Value *tmp = llvm::ConstantInt::get(llvm_index_type, 0);
      for (int j = 0; j < max_num_indices; j++) {
        auto e = stack[i]->extractors[j];
        int b = e.num_bits;
        if (b) {
          if (e.num_bits == e.start || max_num_indices != 1) {
            /*
            emit("tmp = (tmp << {}) + ((i{} >> {}) & ((1 << {}) - 1));",
                 e.num_bits, j, e.start, e.num_bits);
            */
            uint32 mask = (1u << e.num_bits) - 1;
            tmp = builder.CreateShl(tmp, e.num_bits);
            auto patch = builder.CreateAShr(args[j + 1], e.start);
            patch = builder.CreateAnd(patch, mask);
            tmp = builder.CreateAdd(tmp, patch);
          } else {
            TC_WARN("Emitting shortcut indexing");
            emit("tmp = i{};", j);
          }
        }
      }
      bool force_activate = mode == mode_strong_access;
      if (mode != mode_activate) {
        if (force_activate)
          emit("#if 1");
        else
          emit("#if defined(TC_STRUCT)");
      }
      if (stack[i]->type != SNodeType::place) {
        if (mode == mode_query) {
          if (stack[i]->need_activation())
            emit("if (!n{}->is_active(tmp)) return false;", i);
        } else {
          emit("n{}->activate(tmp, {{i0, i1, i2, i3}});", i);
        }
      }
      if (mode != mode_activate) {
        emit("#endif");
      }
      if (mode == mode_weak_access) {
        if (stack[i]->has_null()) {
          emit("if (!n{}->is_active(tmp))", i);
          emit(
              "#if __CUDA_ARCH__\n return "
              "({} *)Managers::get_zeros();\n #else \n return &{}_ambient;\n "
              "#endif",
              snode.node_type_name, snode.node_type_name, snode.node_type_name);
        }
      }
      emit("auto n{} = access_{}(n{}, tmp);", i + 1,
           stack[i + 1]->node_type_name, i);
      node = builder.CreateCall(chain_accessors[stack[i + 1]], {node, tmp});
    }
    if (mode == mode_query) {
      emit("return true;");
    } else {
      emit("return n{};", (int)stack.size() - 1);
    }
    emit("}}");
    emit("");
    // node = builder.
    builder.CreateRet(node);

    TC_WARN_IF(llvm::verifyFunction(*accessor, &errs()),
               "function verification failed");
    leaf_accessors[&snode] = accessor;
    leaf_accessor_names[&snode] = (std::string)accessor->getName();
  }

  for (auto ch : snode.ch) {
    generate_leaf_accessors(*ch);
  }

  stack.pop_back();
}

void StructCompilerLLVM::load_accessors(SNode &snode) {
  for (auto ch : snode.ch) {
    load_accessors(*ch);
  }
  if (snode.type == SNodeType::place) {
    llvm::ExitOnError exit_on_err;
    std::string name = leaf_accessor_names[&snode];
    snode.access_func = (SNode::AccessorFunction)(
        exit_on_err(tlctx->jit->lookup(name)).getAddress());
  } else {
    // snode.stat_func = load_function<SNode::StatFunction>(
    // fmt::format("stat_{}", snode.node_type_name));
  }
  if (snode.has_null()) {
    // snode.clear_func = load_function<SNode::ClearFunction>(
    //    fmt::format("clear_{}", snode.node_type_name));
  }
}

void StructCompilerLLVM::set_parents(SNode &snode) {
  for (auto &c : snode.ch) {
    set_parents(*c);
    c->parent = &snode;
  }
}

void StructCompilerLLVM::run(SNode &node) {
  set_parents(node);
  // bottom to top
  compile(node);

  // get corner coordinates
  /*
  for (int i = 0; i < (int)snodes.size(); i++) {
    auto snode = snodes[i];
    emit(
        "template <> __host__ __device__ void "
        "get_corner_coord<{}>(const "
        "PhysicalIndexGroup &indices, PhysicalIndexGroup &output) {{",
        snode->node_type_name);
    for (int j = 0; j < max_num_indices; j++) {
      auto e = snode->extractors[j];
      emit("output[{}] = indices[{}] & (~((1 << {}) - 1));", j, j,
           e.start + e.num_bits);
    }
    emit("}}");
  }
  */

  // TODO: general allocators
  auto root_size =
      tlctx->jit->getDataLayout().getTypeAllocSize(llvm_types[&root]);

  creator = [=] {
    TC_INFO("Allocating data structure of size {}", root_size);
    return std::malloc(root_size);
  };

  root_type = node.node_type_name;
  generate_leaf_accessors(node);

  TC_INFO("Struct Module IR");
  module->print(errs(), nullptr);

  TC_ASSERT((int)snodes.size() <= max_num_snodes);
  for (int i = 0; i < (int)snodes.size(); i++) {
    // if (snodes[i]->type == SNodeType::pointer ||
    // snodes[i]->type == SNodeType::hashed) {
    if (snodes[i]->type != SNodeType::place) {
      emit(
          "Managers::get<{}>() = "
          "create_unified<SNodeManager<{}>>();",
          snodes[i]->node_type_name, snodes[i]->node_type_name);
    }
  }

  if (get_current_program().config.arch == Arch::gpu) {
    for (int i = 0; i < (int)ambient_snodes.size(); i++) {
      emit("{{");
      auto ntn = ambient_snodes[i]->node_type_name;
      emit("auto ambient_ptr = create_unified<{}::child_type>();", ntn);
      emit("Managers::get_allocator<{}>()->ambient = ambient_ptr;", ntn);
      emit("}}");
    }
  }

  module->setDataLayout(tlctx->jit->getDataLayout());

  tlctx->set_struct_module(module);

  llvm::cantFail(tlctx->jit->addModule(std::move(module)));

  load_accessors(node);
}

std::unique_ptr<StructCompiler> StructCompiler::make(bool use_llvm) {
  if (use_llvm) {
    return std::make_unique<StructCompilerLLVM>();
  } else {
    return std::make_unique<StructCompiler>();
  }
}

TLANG_NAMESPACE_END
