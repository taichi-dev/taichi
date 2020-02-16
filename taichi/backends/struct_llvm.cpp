// Codegen for the hierarchical data structure (LLVM)

#include "struct_llvm.h"
#include "../ir.h"
#include "../program.h"
#include "../unified_allocator.h"
#include "struct.h"
#include "llvm/IR/Verifier.h"
#include <llvm/IR/IRBuilder.h>

TLANG_NAMESPACE_BEGIN

void assert_failed_host(const char *msg) {
  TI_ERROR("Assertion failure: {}", msg);
}

StructCompilerLLVM::StructCompilerLLVM(Program *prog, Arch arch)
    : StructCompiler(prog),
      ModuleBuilder(prog->get_llvm_context(arch)->get_init_module()),
      arch(arch) {
  creator = [] {
    TI_WARN("Data structure creation not implemented");
    return nullptr;
  };
  tlctx = prog->get_llvm_context(arch);
  llvm_ctx = tlctx->ctx.get();
}

void *taichi_allocate_aligned(Program *prog,
                              std::size_t size,
                              std::size_t alignment) {
  return prog->memory_pool->allocate(size, alignment);
}

void StructCompilerLLVM::generate_types(SNode &snode) {
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
  if (type == SNodeType::dense) {
    TI_ASSERT(snode._morton == false);
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
    } else if (snode.dt == DataType::i64) {
      body_type = llvm::Type::getInt64Ty(*ctx);
    } else if (snode.dt == DataType::f64){
      body_type = llvm::Type::getDoubleTy(*ctx);
    } else {
      TI_NOT_IMPLEMENTED
    }
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
    snode.has_aux_structure = true;
  } else {
    llvm_type = body_type;
    snode.has_aux_structure = false;
  }

  TI_ASSERT(llvm_type != nullptr);
  snode_attr[snode].llvm_type = llvm_type;
  snode_attr[snode].llvm_aux_type = aux_type;
  snode_attr[snode].llvm_body_type = body_type;
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
    auto in = call(&builder, "PhysicalCoordinates_get_val", inp_coords,
                   tlctx->get_constant(i));
    auto added = builder.CreateOr(in, addition);
    call(&builder, "PhysicalCoordinates_set_val", outp_coords,
         tlctx->get_constant(i), added);
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

    auto inp_type =
        llvm::PointerType::get(snode_attr[parent].llvm_element_type, 0);
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
    ret = builder.CreateGEP(
        builder.CreateBitCast(args[0], inp_type),
        {tlctx->get_constant(0), tlctx->get_constant(parent->child_id(&snode))},
        "getch");

    builder.CreateRet(
        builder.CreateBitCast(ret, llvm::Type::getInt8PtrTy(*llvm_ctx)));
  }

  // SNode::place & indirect
  // emit end2end accessors for leaf (place) nodes, using chain accessors
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

void StructCompilerLLVM::load_accessors(SNode &snode) {
}

void StructCompilerLLVM::run(SNode &root, bool host) {
  // bottom to top
  collect_snodes(root);

  if (host)
    infer_snode_properties(root);

  auto snodes_rev = snodes;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  for (auto &n : snodes_rev)
    generate_types(*n);

  // TODO: general allocators

  root_type = root.node_type_name;
  generate_leaf_accessors(root);

  if (prog->config.print_struct_llvm_ir) {
    TI_INFO("Struct Module IR");
    module->print(errs(), nullptr);
  }

  TI_ASSERT((int)snodes.size() <= max_num_snodes);

  auto root_size =
      tlctx->jit->getDataLayout().getTypeAllocSize(snode_attr[root].llvm_type);

  module->setDataLayout(tlctx->jit->getDataLayout());

  tlctx->set_struct_module(module);

  if (arch == Arch::x86_64)  // Do not compile the GPU struct module alone since
                             // it's useless unless used with kernels
    tlctx->jit->addModule(std::move(module));

  if (host) {
    for (auto n : snodes) {
      load_accessors(*n);
    }

    // TODO(yuanming-hu): move runtime initialization to somewhere else
    auto initialize_runtime = tlctx->lookup_function<void *(
        void *, void *, int, std::size_t, void *, bool)>("Runtime_initialize");

    auto initialize_runtime2 =
        tlctx->lookup_function<void(void *, void *, int, int)>(
            "Runtime_initialize2");

    auto set_assert_failed = tlctx->lookup_function<void(void *, void *)>(
        "Runtime_set_assert_failed");

    auto allocate_ambient =
        tlctx->lookup_function<void(void *, int)>("Runtime_allocate_ambient");

    auto initialize_allocator =
        tlctx->lookup_function<void *(void *, int, std::size_t)>(
            "NodeAllocator_initialize");

    auto runtime_initialize_thread_pool =
        tlctx->lookup_function<void(void *, void *, void *)>(
            "Runtime_initialize_thread_pool");

    auto runtime_set_root =
        tlctx->lookup_function<void(void *, void *)>("Runtime_set_root");

    // By the time when creator is called, "this" is already destoried. Therefor
    // it is necessary to capture members by values.
    auto snodes = this->snodes;
    auto tlctx = this->tlctx;
    auto root_id = root.id;
    auto prog = this->prog;
    creator = [=]() {
      TI_TRACE("Allocating data structure of size {} B", root_size);
      auto root = initialize_runtime(
          &prog->llvm_runtime, prog, (int)snodes.size(), root_size,
          (void *)&taichi_allocate_aligned, logger.get_level() <= 1);

      auto mem_req_queue = tlctx->lookup_function<void *(void *)>(
          "Runtime_get_mem_req_queue")(prog->llvm_runtime);
      prog->memory_pool->set_queue((MemRequestQueue *)mem_req_queue);

      initialize_runtime2(prog->llvm_runtime, root, root_id,
                          (int)snodes.size());
      for (int i = 0; i < (int)snodes.size(); i++) {
        if (snodes[i]->type == SNodeType::pointer ||
            snodes[i]->type == SNodeType::dynamic) {
          std::size_t node_size;
          if (snodes[i]->type == SNodeType::pointer)
            node_size =
                tlctx->get_type_size(snode_attr[snodes[i]].llvm_element_type);
          else {
            // dynamic. Allocators are for the chunks
            node_size =
                sizeof(void *) +
                tlctx->get_type_size(snode_attr[snodes[i]].llvm_element_type) *
                    snodes[i]->chunk_size;
          }
          TI_TRACE("Initializing allocator for snode {} (node size {})",
                  snodes[i]->id, node_size);
          auto rt = prog->llvm_runtime;
          initialize_allocator(rt, i, node_size);
          TI_TRACE("Allocating ambient element for snode {} (node size {})",
                  snodes[i]->id, node_size);
          allocate_ambient(rt, i);
        }
      }

      runtime_initialize_thread_pool(prog->llvm_runtime, &prog->thread_pool,
                                     (void *)ThreadPool::static_run);
      set_assert_failed(prog->llvm_runtime, (void *)assert_failed_host);

      runtime_set_root(prog->llvm_runtime, root);

      tlctx->lookup_function<void(void *, void *)>("Runtime_set_profiler")(
          prog->llvm_runtime, prog->profiler_llvm.get());

      tlctx->lookup_function<void(void *, void *)>(
          "Runtime_set_profiler_start")(prog->llvm_runtime,
                                        (void *)&ProfilerBase::profiler_start);
      tlctx->lookup_function<void(void *, void *)>("Runtime_set_profiler_stop")(
          prog->llvm_runtime, (void *)&ProfilerBase::profiler_stop);
    };
  }
  tlctx->snode_attr = snode_attr;
}

std::unique_ptr<StructCompiler> StructCompiler::make(bool use_llvm,
                                                     Program *prog,
                                                     Arch arch) {
  if (use_llvm) {
    return std::make_unique<StructCompilerLLVM>(prog, arch);
  } else {
    return std::make_unique<StructCompiler>(prog);
  }
}

bool SNode::need_activation() const {
  return type == SNodeType::pointer || type == SNodeType::hash ||
         (type == SNodeType::dense && _bitmasked) || type == SNodeType::dynamic;
}

TLANG_NAMESPACE_END
