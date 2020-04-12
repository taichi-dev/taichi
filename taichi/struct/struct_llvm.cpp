// Codegen for the hierarchical data structure (LLVM)

#include "struct_llvm.h"
#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "struct.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRBuilder.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

StructCompilerLLVM::StructCompilerLLVM(Program *prog, Arch arch)
    : StructCompiler(prog),
      ModuleBuilder(prog->get_llvm_context(arch)->get_init_module()),
      arch(arch) {
  tlctx = prog->get_llvm_context(arch);
  llvm_ctx = tlctx->ctx.get();
}

void StructCompilerLLVM::generate_types(SNode &snode) {
  TI_AUTO_PROF;
  auto type = snode.type;
  llvm::Type *llvm_type = nullptr;

  auto ctx = llvm_ctx;

  // create children type that supports forking...

  std::vector<llvm::Type *> ch_types;
  for (int i = 0; i < snode.ch.size(); i++) {
    auto ch = snode_attr[snode.ch[i]].llvm_type;
    ch_types.push_back(ch);
  }

  auto ch_type =
      llvm::StructType::create(*ctx, ch_types, snode.node_type_name + "_ch");
  ch_type->setName(snode.node_type_name + "_ch");

  snode_attr[snode].llvm_element_type = ch_type;

  llvm::Type *body_type = nullptr, *aux_type = nullptr;
  if (type == SNodeType::dense || type == SNodeType::bitmasked) {
    TI_ASSERT(snode._morton == false);
    body_type = llvm::ArrayType::get(ch_type, snode.max_num_elements());
    if (type == SNodeType::bitmasked) {
      aux_type = llvm::ArrayType::get(llvm::Type::getInt32Ty(*llvm_ctx),
                                      (snode.max_num_elements() + 31) / 32);
    }
  } else if (type == SNodeType::root) {
    body_type = ch_type;
  } else if (type == SNodeType::place) {
    body_type = tlctx->get_data_type(snode.dt);
  } else if (type == SNodeType::pointer) {
    // mutex
    aux_type = llvm::ArrayType::get(llvm::PointerType::getInt64Ty(*ctx),
                                    snode.max_num_elements());
    body_type = llvm::ArrayType::get(llvm::PointerType::getInt8PtrTy(*ctx),
                                     snode.max_num_elements());
  } else if (type == SNodeType::dynamic) {
    // mutex and n (number of elements)
    aux_type =
        llvm::StructType::get(*ctx, {llvm::PointerType::getInt32Ty(*ctx),
                                     llvm::PointerType::getInt32Ty(*ctx)});
    body_type = llvm::PointerType::getInt8PtrTy(*ctx);
  } else {
    TI_P(snode.type_name());
    TI_NOT_IMPLEMENTED;
  }
  if (aux_type != nullptr) {
    llvm_type = llvm::StructType::create(*ctx, {aux_type, body_type}, "");
  } else {
    llvm_type = body_type;
  }

  TI_ASSERT(llvm_type != nullptr);
  snode_attr[snode].llvm_type = llvm_type;
  snode_attr[snode].llvm_aux_type = aux_type;
  snode_attr[snode].llvm_body_type = body_type;
}

void StructCompilerLLVM::generate_refine_coordinates(SNode *snode) {
  TI_AUTO_PROF;
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

  for (int i = 0; i < taichi_max_num_indices; i++) {
    auto addition = tlctx->get_constant(0);
    if (snode->extractors[i].num_bits) {
      auto mask = ((1 << snode->extractors[i].num_bits) - 1);
      addition = builder.CreateAnd(
          builder.CreateAShr(l, snode->extractors[i].acc_offset), mask);
      addition = builder.CreateShl(
          addition, tlctx->get_constant(snode->extractors[i].start));
    }
    auto in = call(&builder, "PhysicalCoordinates_get_val", inp_coords,
                   tlctx->get_constant(i));
    auto added = builder.CreateOr(in, addition);
    call(&builder, "PhysicalCoordinates_set_val", outp_coords,
         tlctx->get_constant(i), added);
  }
  builder.CreateRetVoid();
}

void StructCompilerLLVM::generate_child_accessors(SNode &snode) {
  TI_AUTO_PROF;
  auto type = snode.type;
  stack.push_back(&snode);

  bool is_leaf = type == SNodeType::place;

  if (!is_leaf) {
    generate_refine_coordinates(&snode);
  }

  if (snode.parent != nullptr) {
    // create the get ch function
    auto parent = snode.parent;

    auto inp_type =
        llvm::PointerType::get(snode_attr[parent].llvm_element_type, 0);

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
    ret = builder.CreateGEP(
        builder.CreateBitCast(args[0], inp_type),
        {tlctx->get_constant(0), tlctx->get_constant(parent->child_id(&snode))},
        "getch");

    builder.CreateRet(
        builder.CreateBitCast(ret, llvm::Type::getInt8PtrTy(*llvm_ctx)));
  }

  for (auto &ch : snode.ch) {
    generate_child_accessors(*ch);
  }

  stack.pop_back();
}

void StructCompilerLLVM::run(SNode &root, bool host) {
  TI_AUTO_PROF;
  // bottom to top
  collect_snodes(root);

  if (host)
    infer_snode_properties(root);

  auto snodes_rev = snodes;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  for (auto &n : snodes_rev)
    generate_types(*n);

  generate_child_accessors(root);

  if (prog->config.print_struct_llvm_ir) {
    TI_INFO("Struct Module IR");
    module->print(errs(), nullptr);
  }

  TI_ASSERT((int)snodes.size() <= taichi_max_num_snodes);

  root_size =
      tlctx->get_data_layout().getTypeAllocSize(snode_attr[root].llvm_type);

  tlctx->set_struct_module(module);
  tlctx->snode_attr = snode_attr;
}

std::unique_ptr<StructCompiler> StructCompiler::make(Program *prog, Arch arch) {
  return std::make_unique<StructCompilerLLVM>(prog, arch);
}

bool SNode::need_activation() const {
  return type == SNodeType::pointer || type == SNodeType::hash ||
         type == SNodeType::bitmasked || type == SNodeType::dynamic;
}

TLANG_NAMESPACE_END
