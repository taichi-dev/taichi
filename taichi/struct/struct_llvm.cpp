#include "taichi/struct/struct_llvm.h"

#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRBuilder.h"

#include "taichi/ir/ir.h"
#include "taichi/struct/struct.h"
#include "taichi/program/program.h"
#include "taichi/util/file_sequence_writer.h"

namespace taichi {
namespace lang {

StructCompilerLLVM::StructCompilerLLVM(Arch arch,
                                       const CompileConfig *config,
                                       TaichiLLVMContext *tlctx,
                                       std::unique_ptr<llvm::Module> &&module)
    : LLVMModuleBuilder(std::move(module), tlctx),
      arch_(arch),
      config_(config),
      tlctx_(tlctx),
      llvm_ctx_(tlctx_->get_this_thread_context()) {
}

StructCompilerLLVM::StructCompilerLLVM(Arch arch,
                                       Program *prog,
                                       std::unique_ptr<llvm::Module> &&module)
    : StructCompilerLLVM(arch,
                         &(prog->config),
                         prog->get_llvm_context(arch),
                         std::move(module)) {
}

void StructCompilerLLVM::generate_types(SNode &snode) {
  TI_AUTO_PROF;
  auto type = snode.type;
  if (snode.is_bit_level)
    return;
  llvm::Type *node_type = nullptr;

  auto ctx = llvm_ctx_;
  TI_ASSERT(ctx == tlctx_->get_this_thread_context());

  // create children type that supports forking...

  std::vector<llvm::Type *> ch_types;
  for (int i = 0; i < snode.ch.size(); i++) {
    if (!snode.ch[i]->is_bit_level) {
      // Bit-level SNodes do not really have a corresponding LLVM type
      auto ch = get_llvm_node_type(module.get(), snode.ch[i].get());
      ch_types.push_back(ch);
    }
  }

  auto ch_type =
      llvm::StructType::create(*ctx, ch_types, snode.node_type_name + "_ch");

  snode.cell_size_bytes = tlctx_->get_type_size(ch_type);

  llvm::Type *body_type = nullptr, *aux_type = nullptr;
  if (type == SNodeType::dense || type == SNodeType::bitmasked) {
    TI_ASSERT(snode._morton == false);
    body_type = llvm::ArrayType::get(ch_type, snode.max_num_elements());
    if (type == SNodeType::bitmasked) {
      aux_type = llvm::ArrayType::get(llvm::Type::getInt32Ty(*llvm_ctx_),
                                      (snode.max_num_elements() + 31) / 32);
    }
  } else if (type == SNodeType::root) {
    body_type = ch_type;
  } else if (type == SNodeType::place) {
    body_type = tlctx_->get_data_type(snode.dt);
  } else if (type == SNodeType::bit_struct) {
    // Generate the bit_struct type
    std::vector<Type *> ch_types;
    std::vector<int> ch_offsets;
    int total_offset = 0;
    for (int i = 0; i < snode.ch.size(); i++) {
      auto &ch = snode.ch[i];
      ch_types.push_back(ch->dt);
      ch_offsets.push_back(total_offset);
      CustomIntType *component_cit = nullptr;
      if (auto cit = ch->dt->cast<CustomIntType>()) {
        component_cit = cit;
      } else if (auto cft = ch->dt->cast<CustomFloatType>()) {
        component_cit = cft->get_digits_type()->as<CustomIntType>();
      } else {
        TI_ERROR("Type {} not supported.", ch->dt->to_string());
      }
      component_cit->set_physical_type(snode.physical_type);
      if (!arch_is_cpu(arch_)) {
        TI_ERROR_IF(data_type_bits(snode.physical_type) < 32,
                    "bit_struct physical type must be at least 32 bits on "
                    "non-CPU backends.");
      }
      ch->bit_offset = total_offset;
      total_offset += component_cit->get_num_bits();
      auto bit_struct_size = data_type_bits(snode.physical_type);
      TI_ERROR_IF(total_offset > bit_struct_size,
                  "Bit struct overflows: {} bits used out of {}.", total_offset,
                  bit_struct_size);
    }

    snode.dt = TypeFactory::get_instance().get_bit_struct_type(
        snode.physical_type, ch_types, ch_offsets);

    DataType container_primitive_type(snode.physical_type);
    body_type = tlctx_->get_data_type(container_primitive_type);
  } else if (type == SNodeType::bit_array) {
    // A bit array SNode should have only one child
    TI_ASSERT(snode.ch.size() == 1);
    auto &ch = snode.ch[0];
    Type *ch_type = ch->dt;
    ch->dt->as<CustomIntType>()->set_physical_type(snode.physical_type);
    if (!arch_is_cpu(arch_)) {
      TI_ERROR_IF(data_type_bits(snode.physical_type) <= 16,
                  "bit_array physical type must be at least 32 bits on "
                  "non-CPU backends.");
    }
    snode.dt = TypeFactory::get_instance().get_bit_array_type(
        snode.physical_type, ch_type, snode.n);

    DataType container_primitive_type(snode.physical_type);
    body_type = tlctx_->get_data_type(container_primitive_type);
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
    node_type = llvm::StructType::create(*ctx, {aux_type, body_type}, "");
  } else {
    node_type = body_type;
  }

  TI_ASSERT(node_type != nullptr);
  TI_ASSERT(body_type != nullptr);

  // Here we create a stub holding 4 LLVM types as struct members.
  // The aim is to give a **unique** name to the stub, so that we can look up
  // these types using this name. This decouples them from the LLVM context.
  // Note that body_type might not have a unique name, since literal structs
  // (such as {i32, i32}) cannot be aliased in LLVM.
  auto stub = llvm::StructType::create(
      *ctx,
      {node_type, body_type, aux_type ? aux_type : llvm::Type::getInt8Ty(*ctx),
       // aux_type might be null
       ch_type},
      type_stub_name(&snode));

  // Create a dummy function in the module with the type stub as return type
  // so that the type is referenced in the module
  auto ft = llvm::FunctionType::get(llvm::PointerType::get(stub, 0), false);
  llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                         type_stub_name(&snode) + "_func", module.get());
}

void StructCompilerLLVM::generate_refine_coordinates(SNode *snode) {
  TI_AUTO_PROF;
  auto coord_type = get_runtime_type("PhysicalCoordinates");
  auto coord_type_ptr = llvm::PointerType::get(coord_type, 0);

  auto ft = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*llvm_ctx_),
      {coord_type_ptr, coord_type_ptr, llvm::Type::getInt32Ty(*llvm_ctx_)},
      false);

  auto func =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                             snode->refine_coordinates_func_name(), *module);

  auto bb = llvm::BasicBlock::Create(*llvm_ctx_, "entry", func);

  llvm::IRBuilder<> builder(bb, bb->begin());
  std::vector<llvm::Value *> args;

  for (auto &arg : func->args()) {
    args.push_back(&arg);
  }

  auto inp_coords = args[0];
  auto outp_coords = args[1];
  auto l = args[2];

  for (int i = 0; i < taichi_max_num_indices; i++) {
    auto addition = tlctx_->get_constant(0);
    if (snode->extractors[i].num_bits) {
      auto mask = ((1 << snode->extractors[i].num_bits) - 1);
      addition = builder.CreateAnd(
          builder.CreateAShr(l, snode->extractors[i].acc_offset), mask);
    }
    auto in = call(&builder, "PhysicalCoordinates_get_val", inp_coords,
                   tlctx_->get_constant(i));
    in = builder.CreateShl(in,
                           tlctx_->get_constant(snode->extractors[i].num_bits));
    auto added = builder.CreateOr(in, addition);
    call(&builder, "PhysicalCoordinates_set_val", outp_coords,
         tlctx_->get_constant(i), added);
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
        llvm::PointerType::get(get_llvm_element_type(module.get(), parent), 0);

    auto ft =
        llvm::FunctionType::get(llvm::Type::getInt8PtrTy(*llvm_ctx_),
                                {llvm::Type::getInt8PtrTy(*llvm_ctx_)}, false);

    auto func =
        llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                               snode.get_ch_from_parent_func_name(), *module);

    auto bb = llvm::BasicBlock::Create(*llvm_ctx_, "entry", func);

    llvm::IRBuilder<> builder(bb, bb->begin());
    std::vector<llvm::Value *> args;

    for (auto &arg : func->args()) {
      args.push_back(&arg);
    }
    llvm::Value *ret;
    ret = builder.CreateGEP(builder.CreateBitCast(args[0], inp_type),
                            {tlctx_->get_constant(0),
                             tlctx_->get_constant(parent->child_id(&snode))},
                            "getch");

    builder.CreateRet(
        builder.CreateBitCast(ret, llvm::Type::getInt8PtrTy(*llvm_ctx_)));
  }

  for (auto &ch : snode.ch) {
    if (!ch->is_bit_level)
      generate_child_accessors(*ch);
  }

  stack.pop_back();
}

std::string StructCompilerLLVM::type_stub_name(SNode *snode) {
  return snode->node_type_name + "_type_stubs";
}

void StructCompilerLLVM::run(SNode &root) {
  TI_AUTO_PROF;
  // bottom to top
  collect_snodes(root);

  auto snodes_rev = snodes;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  for (auto &n : snodes_rev)
    generate_types(*n);

  generate_child_accessors(root);

  if (config_->print_struct_llvm_ir) {
    static FileSequenceWriter writer("taichi_struct_llvm_ir_{:04d}.ll",
                                     "struct LLVM IR");
    writer.write(module.get());
  }

  TI_ASSERT((int)snodes.size() <= taichi_max_num_snodes);

  auto node_type = get_llvm_node_type(module.get(), &root);
  root_size = tlctx_->get_data_layout().getTypeAllocSize(node_type);

  tlctx_->set_struct_module(module);
}

llvm::Type *StructCompilerLLVM::get_stub(llvm::Module *module,
                                         SNode *snode,
                                         uint32 index) {
  TI_ASSERT(module);
  TI_ASSERT(snode);
  auto stub = module->getTypeByName(type_stub_name(snode));
  TI_ASSERT(stub);
  TI_ASSERT(stub->getStructNumElements() == 4);
  TI_ASSERT(0 <= index && index < 4);
  auto type = stub->getContainedType(index);
  TI_ASSERT(type);
  return type;
}

llvm::Type *StructCompilerLLVM::get_llvm_node_type(llvm::Module *module,
                                                   SNode *snode) {
  return get_stub(module, snode, 0);
}

llvm::Type *StructCompilerLLVM::get_llvm_body_type(llvm::Module *module,
                                                   SNode *snode) {
  return get_stub(module, snode, 1);
}

llvm::Type *StructCompilerLLVM::get_llvm_aux_type(llvm::Module *module,
                                                  SNode *snode) {
  return get_stub(module, snode, 2);
}

llvm::Type *StructCompilerLLVM::get_llvm_element_type(llvm::Module *module,
                                                      SNode *snode) {
  return get_stub(module, snode, 3);
}

}  // namespace lang
}  // namespace taichi
