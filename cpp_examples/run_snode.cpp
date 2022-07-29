#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

void run_snode() {
  /*
  import taichi as ti, numpy as np
  ti.init()
  #ti.init(print_ir = True)

  n = 10
  place = ti.field(dtype = ti.i32)
  ti.root.pointer(ti.i, n).place(place)

  @ti.kernel
  def init():
      for index in range(n):
          place[index] = index

  @ti.kernel
  def ret() -> ti.i32:
      sum = 0
      for index in place:
          sum = sum + place[index]
      return sum

  @ti.kernel
  def ext(ext_arr: ti.ext_arr()):
      for index in place:
          ext_arr[index] = place[index]

  init()
  print(ret())
  ext_arr = np.zeros(n, np.int32)
  ext(ext_arr)
  #ext_arr = place.to_numpy()
  print(ext_arr)
  */

  using namespace taichi;
  using namespace lang;
  auto program = Program(Arch::x64);
  /*CompileConfig config_print_ir;
  config_print_ir.print_ir = true;
  prog_.config = config_print_ir;*/  // print_ir = True

  int n = 10;
  program.materialize_runtime();
  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->pointer(Axis(0), n, false);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/false);

  std::unique_ptr<Kernel> kernel_init, kernel_ret, kernel_ext;

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
    auto *loop = builder.create_range_for(zero, n_stmt, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, index);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  {
    /*
    @ti.kernel
    def ret():
      sum = 0
      for index in place:
        sum = sum + place[index];
      return sum
    */
    IRBuilder builder;
    auto *sum = builder.create_local_var(PrimitiveType::i32);
    auto *loop = builder.create_struct_for(pointer, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *sum_old = builder.create_local_load(sum);
      auto *place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");
  }

  {
    /*
    @ti.kernel
    def ext(ext: ti.ext_arr()):
      for index in place:
        ext[index] = place[index];
    # ext = place.to_numpy()
    */
    IRBuilder builder;
    auto *loop = builder.create_struct_for(pointer, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ext = builder.create_external_ptr(
          builder.create_arg_load(0, PrimitiveType::i32, true), {index});
      auto *place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_global_store(ext, place_index);
    }

    kernel_ext = std::make_unique<Kernel>(program, builder.extract_ir(), "ext");
    kernel_ext->insert_arg(get_data_type<int>(), true);
  }

  auto ctx_init = kernel_init->make_launch_context();
  auto ctx_ret = kernel_ret->make_launch_context();
  auto ctx_ext = kernel_ext->make_launch_context();
  std::vector<int> ext_arr(n);
  ctx_ext.set_arg_external_array_with_shape(0, taichi::uint64(ext_arr.data()),
                                            n, {n});

  (*kernel_init)(ctx_init);
  (*kernel_ret)(ctx_ret);
  std::cout << program.fetch_result<int>(0) << std::endl;
  (*kernel_ext)(ctx_ext);
  for (int i = 0; i < n; i++)
    std::cout << ext_arr[i] << " ";
  std::cout << std::endl;
}
