#include "gtest/gtest.h"

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

TEST(Meshfor, Basic) {
  auto program = Program(Arch::cuda);
  CompileConfig config_print_ir(program.config);
  config_print_ir.print_ir = true;
  program.config = config_print_ir;  // print_ir = True
  program.materialize_runtime();
  int n = 10;
  // auto *root = program.get_snode_root(SNodeTree::kFirstID);
  auto *root = new SNode(0, SNodeType::root);
  auto *dense = &(root->dense(0, n, false));
  auto *place = &dense->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root));
  std::unique_ptr<Kernel> kernel_init, kernel_mesh;
  {
    /*
    @ti.kernel
    def init():
      for index in range(n):
        place[index] = index
    */
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, builder.create_mul(index, index));
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }
  {
    /*
    @ti.kernel
    def mesh():
      for index in mesh:
        print(index)
    */
    IRBuilder builder;
    auto *loop = builder.create_mesh_for(place, 32);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      builder.create_print(index, "\n");
    }

    kernel_mesh =
        std::make_unique<Kernel>(program, builder.extract_ir(), "mesh");
  }
  auto ctx_init = kernel_init->make_launch_context();
  auto ctx_mesh = kernel_mesh->make_launch_context();
  (*kernel_init)(ctx_init);
  (*kernel_mesh)(ctx_mesh);
}
}  // namespace lang
}  // namespace taichi
