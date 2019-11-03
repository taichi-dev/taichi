// Codegen for the hierarchical data structure (LLVM)

#include "struct_llvm.h"
#include "../ir.h"
#include "../program.h"
#include "struct.h"
#include "llvm/IR/Verifier.h"
#include <llvm/IR/IRBuilder.h>

TLANG_NAMESPACE_BEGIN

StructCompilerLLVM::StructCompilerLLVM(Arch arch)
    : StructCompiler(),
      ModuleBuilder(
          get_current_program().get_llvm_context(arch)->get_init_module()),
      arch(arch) {
  creator = [] {
    TC_WARN("Data structure creation not implemented");
    return nullptr;
  };
  tlctx = get_current_program().get_llvm_context(arch);
  llvm_ctx = tlctx->ctx.get();
}

void StructCompilerLLVM::generate_types(SNode &snode) {
  auto type = snode.type;
  llvm::Type *llvm_type = nullptr;

  auto ctx = llvm_ctx;

  // create children type that supports forking...

  std::vector<llvm::Type *> ch_types;
  for (int i = 0; i < snode.ch.size(); i++) {
    auto ch = snode.ch[i]->llvm_type;
    ch_types.push_back(ch);
  }

  auto ch_type =
      llvm::StructType::create(*ctx, ch_types, snode.node_type_name + "_ch");
  ch_type->setName(snode.node_type_name + "_ch");

  snode.llvm_element_type = ch_type;

  llvm::Type *body_type = nullptr, *aux_type = nullptr;
  if (type == SNodeType::dense) {
    TC_ASSERT(snode._morton == false);
    body_type = llvm::ArrayType::get(ch_type, snode.max_num_elements());
    if (snode._bitmasked) {
      aux_type = llvm::ArrayType::get(Type::getInt32Ty(*llvm_ctx),
                                      (snode.max_num_elements() + 31) / 32);
    }
  } else if (type == SNodeType::root) {
    body_type = ch_type;
  } else if (type == SNodeType::place) {
    if (snode.dt == DataType::f32) {
      body_type = llvm::Type::getFloatTy(*ctx);
    } else if (snode.dt == DataType::i32) {
      body_type = llvm::Type::getInt32Ty(*ctx);
    } else {
      body_type = llvm::Type::getDoubleTy(*ctx);
    }
  } else if (type == SNodeType::pointer) {
    body_type = llvm::PointerType::getInt8PtrTy(*ctx);
    // mutex
    aux_type = llvm::PointerType::getInt32Ty(*ctx);
  } else {
    TC_P(snode.type_name());
    TC_NOT_IMPLEMENTED;
  }
  if (aux_type != nullptr) {
    llvm_type = llvm::StructType::create(*ctx, {body_type, aux_type}, "");
    snode.has_aux_structure = true;
  } else {
    llvm_type = body_type;
    snode.has_aux_structure = false;
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
  snode.llvm_type = llvm_type;
  snode.llvm_body_type = body_type;
  snode.llvm_aux_type = aux_type;
}

void StructCompilerLLVM::generate_refine_coordinates(SNode *snode) {
  auto coord_type = get_runtime_type("PhysicalCoordinates");
  auto coord_type_ptr = llvm::PointerType::get(coord_type, 0);

  auto ft = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*llvm_ctx),
      {coord_type_ptr, coord_type_ptr, llvm::Type::getInt32Ty(*llvm_ctx)},
      false);

  auto func = Function::Create(ft, Function::ExternalLinkage,
                               snode->refine_coordinates_func_name(), *module);

  auto bb = BasicBlock::Create(*llvm_ctx, "entry", func);

  llvm::IRBuilder<> builder(bb, bb->begin());
  std::vector<Value *> args;

  for (auto &arg : func->args()) {
    args.push_back(&arg);
  }

  auto inp_coords = args[0];
  auto outp_coords = args[1];
  auto l = args[2];

  for (int i = 0; i < max_num_indices; i++) {
    auto addition = tlctx->get_constant(0);
    if (snode->extractors[i].num_bits) {
      auto mask = ((1 << snode->extractors[i].num_bits) - 1);
      addition = builder.CreateAnd(
          builder.CreateAShr(l, snode->extractors[i].acc_offset), mask);
      addition = builder.CreateShl(
          addition, tlctx->get_constant(snode->extractors[i].start));
    }
    auto in =
        builder.CreateCall(get_runtime_function("PhysicalCoordinates_get_val"),
                           {inp_coords, tlctx->get_constant(i)});
    auto added = builder.CreateOr(in, addition);
    builder.CreateCall(get_runtime_function("PhysicalCoordinates_set_val"),
                       {outp_coords, tlctx->get_constant(i), added});
  }
  builder.CreateRetVoid();
}

void StructCompilerLLVM::generate_leaf_accessors(SNode &snode) {
  auto type = snode.type;
  stack.push_back(&snode);

  bool is_leaf = type == SNodeType::place;

  if (!is_leaf) {
    generate_refine_coordinates(&snode);
  }

  if (snode.parent != nullptr) {
    // create the get ch function
    auto parent = snode.parent;

    auto inp_type = llvm::PointerType::get(parent->llvm_element_type, 0);
    // auto ret_type = llvm::PointerType::get(snode.llvm_type, 0);

    auto ft =
        llvm::FunctionType::get(llvm::Type::getInt8PtrTy(*llvm_ctx),
                                {llvm::Type::getInt8PtrTy(*llvm_ctx)}, false);

    auto func = Function::Create(ft, Function::ExternalLinkage,
                                 snode.get_ch_from_parent_func_name(), *module);

    auto bb = BasicBlock::Create(*llvm_ctx, "entry", func);

    llvm::IRBuilder<> builder(bb, bb->begin());
    std::vector<Value *> args;

    for (auto &arg : func->args()) {
      args.push_back(&arg);
    }
    llvm::Value *ret;
    ret = builder.CreateGEP(builder.CreateBitCast(args[0], inp_type),
                            {tlctx->get_constant(0), /// tlctx->get_constant(0),
                             tlctx->get_constant(parent->child_id(&snode))},
                            "getch");

    builder.CreateRet(
        builder.CreateBitCast(ret, llvm::Type::getInt8PtrTy(*llvm_ctx)));
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

  for (auto ch : snode.ch) {
    generate_leaf_accessors(*ch);
  }

  stack.pop_back();
}

void StructCompilerLLVM::load_accessors(SNode &snode) {}

void StructCompilerLLVM::run(SNode &root, bool host) {
  // bottom to top
  collect_snodes(root);

  if (host)
    infer_snode_properties(root);

  auto snodes_rev = snodes;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  for (auto &n : snodes_rev)
    generate_types(*n);

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

  root_type = root.node_type_name;
  generate_leaf_accessors(root);

  if (get_current_program().config.print_struct_llvm_ir) {
    TC_INFO("Struct Module IR");
    module->print(errs(), nullptr);
  }

  TC_ASSERT((int)snodes.size() <= max_num_snodes);
  for (int i = 0; i < (int)snodes.size(); i++) {
    // if (snodes[i]->type == SNodeType::pointer ||
    // snodes[i]->type == SNodeType::hashed) {
    if (snodes[i]->type != SNodeType::place) {
      emit("Managers::get<{}>() = "
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

  auto root_size = tlctx->jit->getDataLayout().getTypeAllocSize(root.llvm_type);
  // initializer

  {
    auto ft =
        llvm::FunctionType::get(llvm::Type::getInt8PtrTy(*llvm_ctx),
                                {llvm::Type::getInt8PtrTy(*llvm_ctx)}, false);
    auto init = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                       "initialize_data_structure", *module);
    std::vector<llvm::Value *> args;
    for (auto &arg : init->args()) {
      args.push_back(&arg);
    }
    auto bb = BasicBlock::Create(*llvm_ctx, "body", init);
    llvm::IRBuilder<> builder(bb, bb->begin());
    auto runtime_ty = get_runtime_type("Runtime");
    auto ret = builder.CreateCall(
        get_runtime_function("Runtime_initialize"),
        {builder.CreateBitCast(
             args[0],
             llvm::PointerType::get(llvm::PointerType::get(runtime_ty, 0), 0)),
         tlctx->get_constant((int)snodes.size()),
         tlctx->get_constant(root_size), tlctx->get_constant(root.id)});
    builder.CreateRet(ret);
  }

  module->setDataLayout(tlctx->jit->getDataLayout());

  tlctx->set_struct_module(module);

  if (arch == Arch::x86_64) // Do not compile the GPU struct module alone since
                            // it's useless unless used with kernels
    tlctx->jit->addModule(std::move(module));

  if (host) {
    for (auto n : snodes) {
      load_accessors(*n);
    }

    auto initialize_data_structure =
        tlctx->lookup_function<std::function<void *(void *)>>(
            "initialize_data_structure");

    auto get_allocator =
        tlctx->lookup_function<std::function<void *(void *, int)>>(
            "Runtime_get_node_allocators");

    auto allocate_ambient =
        tlctx->lookup_function<std::function<void(void *, int)>>(
            "Runtime_allocate_ambient");

    auto initialize_allocator =
        tlctx->lookup_function<std::function<void *(void *, std::size_t)>>(
            "NodeAllocator_initialize");

    auto snodes = this->snodes;
    auto tlctx = this->tlctx;
    creator = [=]() {
      TC_INFO("Allocating data structure of size {}", root_size);
      auto root_ptr =
          initialize_data_structure(&get_current_program().llvm_runtime);
      for (int i = 0; i < (int)snodes.size(); i++) {
        if (snodes[i]->type == SNodeType::pointer) {
          auto element_size =
              tlctx->get_type_size(snodes[i]->ch[0]->llvm_body_type);
          TC_INFO("Initializing allocator for snode {} (element size {})",
                  snodes[i]->id, element_size);
          auto rt = get_current_program().llvm_runtime;
          auto allocator = get_allocator(rt, i);
          initialize_allocator(allocator, element_size);
          TC_INFO("Allocating ambient element for snode {} (element size {})",
                  snodes[i]->id, element_size);
          allocate_ambient(rt, i);
        }
      }
      return (void *)root_ptr;
    };
  }
}

std::unique_ptr<StructCompiler> StructCompiler::make(bool use_llvm, Arch arch) {
  if (use_llvm) {
    return std::make_unique<StructCompilerLLVM>(arch);
  } else {
    return std::make_unique<StructCompiler>();
  }
}

llvm::Type *SNode::get_body_type() { return llvm_body_type; }
llvm::Type *SNode::get_aux_type() { return llvm_aux_type; }

TLANG_NAMESPACE_END
