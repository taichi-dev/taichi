#include "gtest/gtest.h"

#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "tests/cpp/program/test_program.h"

namespace taichi::lang {

std::function<void(const std::string &)>
make_pass_printer(bool verbose, const std::string &kernel_name, IRNode *ir) {
  if (!verbose) {
    return [](const std::string &) {};
  }
  return [ir, kernel_name](const std::string &pass) {
    TI_INFO("[{}] {}:", kernel_name, pass);
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  };
}

TEST(Half2Vectorization, Ndarray) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  /*
  Before:
      i32 const_0 = ConstStmt(0)
      i32 const_1 = ConstStmt(1)

      f16* ptr_0 = ExternalPtrStmt(arg, [$1, const_0])
      f16* ptr_1 = ExternalPtrStmt(arg, [$1, const_1])

      f16 old_val0 = AtomicStmt(ptr_0, $7)
      f16 old_val1 = AtomicStmt(ptr_1, $8)

  After:
    TensorType(2, f16) val = MatrixInitStmt([$7, $8])

    TensorType(2, f16)* ptr = ExternalPtrStmt(arg, [$1])
    TensorType(2, f16) old_val = AtomicStmt(ptr, val)

    TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
    StoreStmt(old_val, old_val_alloc)

    f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
    f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

    alloca_stmt0->replace_all_usages_with(old_val0);
    alloca_stmt1->replace_all_usages_with(old_val1);
  */

  auto ret_type =
      TypeFactory::get_instance().get_pointer_type(PrimitiveType::f16);
  std::vector<StructMember> members;
  members.push_back({ret_type, "data_ptr"});
  auto type = TypeFactory::get_instance().get_struct_type(members);

  auto argload_stmt = block->push_back<ArgLoadStmt>(
      0 /*arg_id*/, type, /*is_ptr*/ true, /*is_grad*/ false,
      /*create_load*/ false);
  auto const_0_stmt = block->push_back<ConstStmt>(TypedConstant(0));
  auto const_1_stmt = block->push_back<ConstStmt>(TypedConstant(1));

  auto val_0_stmt = block->push_back<ConstStmt>(TypedConstant(10));
  auto val_1_stmt = block->push_back<ConstStmt>(TypedConstant(20));

  std::vector<Stmt *> external_ptr_indices0 = {const_0_stmt};
  std::vector<Stmt *> external_ptr_indices1 = {const_1_stmt};
  auto external_ptr_stmt_0 =
      block->push_back<ExternalPtrStmt>(argload_stmt, external_ptr_indices0);
  auto external_ptr_stmt_1 =
      block->push_back<ExternalPtrStmt>(argload_stmt, external_ptr_indices1);

  block->push_back<AtomicOpStmt>(AtomicOpType::add, external_ptr_stmt_0,
                                 val_0_stmt);
  block->push_back<AtomicOpStmt>(AtomicOpType::add, external_ptr_stmt_1,
                                 val_1_stmt);

  irpass::type_check(block.get(), CompileConfig());

  /*
    Before:
      <f16> $0 = arg[0]
      <i32> $1 = const 0
      <i32> $2 = const 1
      <i32> $3 = const 10
      <i32> $4 = const 20
      <*f16> $5 = external_ptr $0, [$1] element_dim=14 layout=SOA is_grad=false
      <*f16> $6 = external_ptr $0, [$2] element_dim=64 layout=SOA is_grad=false
      <f16> $7 = cast_value<f16> $3
      <f16> $8 = atomic add($5, $7)
      <f16> $9 = cast_value<f16> $4
      <f16> $10 = atomic add($6, $9)
  */

  irpass::vectorize_half2(block.get());

  /*
  After:
      <f16> $0 = argload[0]
      <i32> $1 = const 0
      <i32> $2 = const 1
      <i32> $3 = const 10
      <i32> $4 = const 20
      <*f16> $5 = external_ptr $0, [$1] element_dim=14 layout=SOA is_grad=false
      <*f16> $6 = external_ptr $0, [$2] element_dim=0 layout=AOS is_grad=false
      <f16> $7 = cast_value<f16> $3
      <i32> $8 = const 0
      <i32> $9 = const 1
      <[Tensor (2) f16]> $10 = [$7, $17]
      <*[Tensor (2) f16]> $11 = external_ptr $0, [], (2) element_dim=-1
  layout=AOS is_grad=false
      <[Tensor (2) f16]> $12 = atomic add($11, $10)
      <*[Tensor (2) f16]> $13 = alloca
      $14 : local store [$13 <- $12]
      <f16> $15 = shift ptr [$13 + $8]
      <f16> $16 = shift ptr [$13 + $9]
      <f16> $17 = cast_value<f16> $4
  */
  EXPECT_EQ(block->size(), 18);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[10]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[11]->is<ExternalPtrStmt>(), true);
  EXPECT_EQ(block->statements[12]->is<AtomicOpStmt>(), true);
}

TEST(Half2Vectorization, GlobalTemporary) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  /*
    Before:
      f16* ptr_0 = GlobalTempStmt(offset0)
      f16* ptr_1 = GlobalTempStmt(offset0 + 2)

      f16 old_val0 = AtomicStmt(ptr_0, $7)
      f16 old_val1 = AtomicStmt(ptr_1, $8)

    After:
      TensorType(2, f16) val = MatrixInitStmt([$7, $8])

      TensorType(2, f16)* ptr = GlobalTempStmt(offset0)
      TensorType(2, f16) old_val = AtomicStmt(ptr, val)

      TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
      StoreStmt(old_val, old_val_alloc)

      f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
      f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

      alloca_stmt0->replace_all_usages_with(old_val0);
      alloca_stmt1->replace_all_usages_with(old_val1);
  */

  auto val_0_stmt = block->push_back<ConstStmt>(TypedConstant(10));
  auto val_1_stmt = block->push_back<ConstStmt>(TypedConstant(20));

  auto global_temp_stmt_0 =
      block->push_back<GlobalTemporaryStmt>(0, PrimitiveType::f16);
  auto global_temp_stmt_1 =
      block->push_back<GlobalTemporaryStmt>(2, PrimitiveType::f16);

  block->push_back<AtomicOpStmt>(AtomicOpType::add, global_temp_stmt_0,
                                 val_0_stmt);
  block->push_back<AtomicOpStmt>(AtomicOpType::add, global_temp_stmt_1,
                                 val_1_stmt);

  irpass::type_check(block.get(), CompileConfig());

  /*
    Before:
      <i32> $0 = const 10
      <i32> $1 = const 20
      <*f16> $2 = global tmp var (offset = 0 B)
      <*f16> $3 = global tmp var (offset = 2 B)
      <f16> $4 = cast_value<f16> $0
      <f16> $5 = atomic add($2, $4)
      <f16> $6 = cast_value<f16> $1
      <f16> $7 = atomic add($3, $6)
  */

  irpass::vectorize_half2(block.get());
  /*
    After:
      <i32> $0 = const 10
      <i32> $1 = const 20
      <*f16> $2 = global tmp var (offset = 0 B)
      <*f16> $3 = global tmp var (offset = 2 B)
      <f16> $4 = cast_value<f16> $0
      <i32> $5 = const 0
      <i32> $6 = const 1
      <[Tensor (2) f16]> $7 = [$4, $14]
      <*[Tensor (2) f16]> $8 = global tmp var (offset = 0 B)
      <[Tensor (2) f16]> $9 = atomic add($8, $7)
      <*[Tensor (2) f16]> $10 = alloca
      $11 : local store [$10 <- $9]
      <f16> $12 = shift ptr [$10 + $5]
      <f16> $13 = shift ptr [$10 + $6]
      <f16> $14 = cast_value<f16> $1
  */
  EXPECT_EQ(block->size(), 15);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[7]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[8]->is<GlobalTemporaryStmt>(), true);
  EXPECT_EQ(block->statements[9]->is<AtomicOpStmt>(), true);
}

TEST(Half2Vectorization, Field) {
  // Basic tests within a basic block
  TestProgram test_prog;
  test_prog.setup();

  auto block = std::make_unique<Block>();

  auto func = []() {};
  auto kernel =
      std::make_unique<Kernel>(*test_prog.prog(), func, "fake_kernel");

  /*
    Before:
        gen* container = SNodeLookupStmt(...)

        fp16* ptr_0 = GetChStmt(container, 0)
        fp16* ptr_1 = GetChStmt(container, 1)

        f16 old_val0 = AtomicStmt(ptr_0, $7)
        f16 old_val1 = AtomicStmt(ptr_1, $8)

    After:
        TensorType(2, f16) val = MatrixInitStmt([$7, $8])

        TensorType(2, f16)* ptr = GetChStmt(container, 0)
        TensorType(2, f16) old_val = AtomicStmt(ptr, val)

        TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
        StoreStmt(old_val, old_val_alloc)

        f16 old_val0 = MatrixPtrStmt(old_val_alloc, 0)
        f16 old_val1 = MatrixPtrStmt(old_val_alloc, 1)

        alloca_stmt0->replace_all_usages_with(old_val0);
        alloca_stmt1->replace_all_usages_with(old_val1);
  */
  auto get_root = block->push_back<GetRootStmt>();
  auto linearized_empty = block->push_back<LinearizeStmt>(std::vector<Stmt *>(),
                                                          std::vector<int>());
  SNode root(0, SNodeType::root);
  root.insert_children(SNodeType::place);
  root.insert_children(SNodeType::place);

  auto lookup = block->push_back<SNodeLookupStmt>(&root, get_root,
                                                  linearized_empty, false);

  auto get_ch_stmt_0 = block->push_back<GetChStmt>(lookup, 0);
  auto get_ch_stmt_1 = block->push_back<GetChStmt>(lookup, 1);

  get_ch_stmt_0->ret_type = PrimitiveType::f16;
  get_ch_stmt_1->ret_type = PrimitiveType::f16;
  get_ch_stmt_0->ret_type.set_is_pointer(true);
  get_ch_stmt_1->ret_type.set_is_pointer(true);
  get_ch_stmt_0->as<GetChStmt>()->overrided_dtype = true;
  get_ch_stmt_1->as<GetChStmt>()->overrided_dtype = true;

  auto val_0_stmt = block->push_back<ConstStmt>(TypedConstant(10));
  auto val_1_stmt = block->push_back<ConstStmt>(TypedConstant(20));

  block->push_back<AtomicOpStmt>(AtomicOpType::add, get_ch_stmt_0, val_0_stmt);
  block->push_back<AtomicOpStmt>(AtomicOpType::add, get_ch_stmt_1, val_1_stmt);

  irpass::type_check(block.get(), CompileConfig());

  /*
    Before:
      <*gen> $0 = get root nullptr
      <i32> $1 = linearized(ind {}, stride {})
      <*gen> $2 = [S1root][root]::lookup($0, $1) activate = false
      <*f16> $3 = get child [S1root->S2place<gen>] $2
      <*f16> $4 = get child [S1root->S3place<gen>] $2
      <i32> $5 = const 10
      <i32> $6 = const 20
      <f16> $7 = cast_value<f16> $5
      <f16> $8 = atomic add($3, $7)
      <f16> $9 = cast_value<f16> $6
      <f16> $10 = atomic add($4, $9)
  */

  irpass::vectorize_half2(block.get());
  /*
    After:
      <*gen> $0 = get root nullptr
      <i32> $1 = linearized(ind {}, stride {})
      <*gen> $2 = [S1root][root]::lookup($0, $1) activate = false
      <*f16> $3 = get child [S1root->S2place<gen>] $2
      <*f16> $4 = get child [S1root->S3place<gen>] $2
      <i32> $5 = const 10
      <i32> $6 = const 20
      <f16> $7 = cast_value<f16> $5
      <i32> $8 = const 0
      <i32> $9 = const 1
      <[Tensor (2) f16]> $10 = [$7, $17]
      <*[Tensor (2) f16]> $11 = get child [S1root->S2place<gen>] $2
      <[Tensor (2) f16]> $12 = atomic add($11, $10)
      <*[Tensor (2) f16]> $13 = alloca
      $14 : local store [$13 <- $12]
      <f16> $15 = shift ptr [$13 + $8]
      <f16> $16 = shift ptr [$13 + $9]
      <f16> $17 = cast_value<f16> $6
  */
  EXPECT_EQ(block->size(), 18);

  // Check for scalarized statements
  EXPECT_EQ(block->statements[10]->is<MatrixInitStmt>(), true);
  EXPECT_EQ(block->statements[11]->is<GetChStmt>(), true);
  EXPECT_EQ(block->statements[12]->is<AtomicOpStmt>(), true);
}

}  // namespace taichi::lang
