#ifdef TI_WITH_LLVM
#include "gtest/gtest.h"

#include <memory>
#include <unordered_set>

#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/BasicBlock.h"

#include "taichi/rhi/arch.h"
#include "taichi/ir/snode.h"
#include "taichi/codegen/llvm/llvm_codegen_utils.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/program.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/struct_llvm.h"

namespace taichi {

namespace lang {
namespace {

constexpr char kFuncName[] = "run_refine_coords";

class InvokeRefineCoordinatesBuilder : public LLVMModuleBuilder {
 public:
  // 1st arg: Value of the first parent physical coordinates
  // 2nd arg: The child index
  // ret    : Value of the first child physical coordinates
  using FuncType = int (*)(int, int);

  static FuncType build(const SNode *snode, TaichiLLVMContext *tlctx) {
    InvokeRefineCoordinatesBuilder mb{tlctx};
    mb.run_jit(snode);
    LLVMCompiledData data;
    data.module = std::move(mb.module);
    data.used_tree_ids = std::move(mb.used_snode_tree_ids);
    data.tasks.emplace_back(kFuncName);
    std::vector<std::unique_ptr<LLVMCompiledData>> data_list;
    data_list.push_back(std::make_unique<LLVMCompiledData>(std::move(data)));
    auto linked_data = tlctx->link_compiled_tasks(std::move(data_list));
    auto *jit = tlctx->create_jit_module(std::move(linked_data->module));
    auto *fn = jit->lookup_function(kFuncName);
    return reinterpret_cast<FuncType>(fn);
  }

 private:
  InvokeRefineCoordinatesBuilder(TaichiLLVMContext *tlctx)
      : LLVMModuleBuilder(tlctx->new_module("kernel"), tlctx) {
    this->llvm_context = this->tlctx->get_this_thread_context();
    this->builder = std::make_unique<llvm::IRBuilder<>>(*llvm_context);
  }

  void run_jit(const SNode *snode) {
    // pseudo code:
    //
    // int run_refine_coords(int parent_coords_first_comp, int child_index) {
    //   PhysicalCoordinates parent_coords;
    //   PhysicalCoordinates child_coords;
    //   parent_coord.val[0] = parent_coords_first_comp;
    //   snode_refine_coordinates(&parent_coords, &child_coords, child_index);
    //   return child_coords.val[0];
    // }
    auto *const int32_ty = llvm::Type::getInt32Ty(*llvm_context);
    auto *const func_ty =
        llvm::FunctionType::get(int32_ty, {int32_ty, int32_ty},
                                /*isVarArg=*/false);
    auto *const func = llvm::Function::Create(
        func_ty, llvm::Function::ExternalLinkage, kFuncName, module.get());
    std::vector<llvm::Value *> args;
    for (auto &a : func->args()) {
      args.push_back(&a);
    }
    auto *const parent_coords_first_component = args[0];
    auto *const child_index = args[1];

    this->entry_block = llvm::BasicBlock::Create(*llvm_context, "entry", func);
    builder->SetInsertPoint(entry_block);

    auto *const index0 = tlctx->get_constant(0);

    RuntimeObject parent_coords{kLLVMPhysicalCoordinatesName, this,
                                builder.get()};
    parent_coords.set("val", index0, parent_coords_first_component);
    auto *refine_fn_struct = tlctx->get_struct_function(
        snode->refine_coordinates_func_name(), snode->get_snode_tree_id());
    auto *refine_fn = llvm::cast<llvm::Function>(
        module
            ->getOrInsertFunction(refine_fn_struct->getName(),
                                  refine_fn_struct->getFunctionType(),
                                  refine_fn_struct->getAttributes())
            .getCallee());
    used_snode_tree_ids.insert(snode->get_snode_tree_id());
    RuntimeObject child_coords{kLLVMPhysicalCoordinatesName, this,
                               builder.get()};
    builder->CreateCall(refine_fn,
                        {parent_coords.ptr, child_coords.ptr, child_index});
    auto *ret_val = child_coords.get("val", index0);
    builder->CreateRet(ret_val);

    llvm::verifyFunction(*func);
  }

  std::unordered_set<int> used_snode_tree_ids;
};

struct BitsRange {
  int begin{0};
  int end{0};

  int extract(int v) const {
    const unsigned mask = (1U << (end - begin)) - 1;
    return (v >> begin) & mask;
  }
};

constexpr int kPointerSize = 5;
constexpr int kDenseSize = 7;

class RefineCoordinatesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    arch_ = host_arch();
    config_.packed = false;
    config_.print_kernel_llvm_ir = false;
    prog_ = std::make_unique<Program>(arch_);
    auto *llvm_prog_ = get_llvm_program(prog_.get());
    tlctx_ = llvm_prog_->get_llvm_context(arch_);

    root_snode_ = std::make_unique<SNode>(/*depth=*/0, /*t=*/SNodeType::root);
    const std::vector<Axis> axes = {Axis{0}};
    ptr_snode_ = &(root_snode_->pointer(axes, kPointerSize, false));
    dense_snode_ = &(ptr_snode_->dense(axes, kDenseSize, false));
    // Must end with a `place` SNode.
    auto &leaf_snode = dense_snode_->insert_children(SNodeType::place);
    leaf_snode.dt = PrimitiveType::f32;

    auto sc = std::make_unique<StructCompilerLLVM>(arch_, &config_, tlctx_,
                                                   tlctx_->new_module("struct"),
                                                   /*snode_tree_id=*/0);
    sc->run(*root_snode_);
  }

  Arch arch_;
  CompileConfig config_;
  // We shouldn't need a Program instance in this test. Unfortunately, a few
  // places depend on the global |current_program|, so we have to.
  // ¯\_(ツ)_/¯
  std::unique_ptr<Program> prog_{nullptr};
  TaichiLLVMContext *tlctx_{nullptr};

  std::unique_ptr<SNode> root_snode_{nullptr};
  SNode *ptr_snode_{nullptr};
  SNode *dense_snode_{nullptr};
};

TEST_F(RefineCoordinatesTest, Basic) {
  auto *refine_ptr_fn =
      InvokeRefineCoordinatesBuilder::build(ptr_snode_, tlctx_);
  auto *refine_dense_fn =
      InvokeRefineCoordinatesBuilder::build(dense_snode_, tlctx_);

  const BitsRange dense_bit_range{/*begin=*/0,
                                  /*end=*/dense_snode_->extractors[0].num_bits};
  const BitsRange ptr_bit_range{
      /*begin=*/dense_bit_range.end,
      /*end=*/dense_bit_range.end + ptr_snode_->extractors[0].num_bits};
  constexpr int kRootPhyCoord = 0;
  for (int i = 0; i < kPointerSize; ++i) {
    const int ptr_phy_coord = refine_ptr_fn(kRootPhyCoord, i);
    for (int j = 0; j < kDenseSize; ++j) {
      const int loop_index = refine_dense_fn(ptr_phy_coord, j);
      // TODO: This is basically doing a lower_scalar_ptr() manually.
      // We should modularize that function, and use it to generate IRs that
      // does the bit extraction procedure.
      const int dense_portion = dense_bit_range.extract(loop_index);
      const int ptr_portion = ptr_bit_range.extract(loop_index);
      EXPECT_EQ(dense_portion, j);
      EXPECT_EQ(ptr_portion, i);
    }
  }
}

}  // namespace
}  // namespace lang
}  // namespace taichi
#endif  // #ifdef TI_WITH_LLVM
