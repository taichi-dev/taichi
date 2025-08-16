#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

int main() {
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
          place[index] = index * 2 + 1

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
  auto program = Program(host_arch());
  // program.get_program_impl()->config->opt_level = 0;
  // program.get_program_impl()->config->external_optimization_level = 0;
  // program.get_program_impl()->config->advanced_optimization = false;
  // program.get_program_impl()->config->print_ir = true;
  const auto &config = program.compile_config();
  /*CompileConfig config_print_ir;
  config_print_ir.print_ir = true;
  prog_.config = config_print_ir;*/  // print_ir = True

  int n = 10;
  program.materialize_runtime();
  auto root = new SNode(0, SNodeType::root);
  auto pointer = &root->pointer(Axis(0), n);
  auto place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/false);

  std::unique_ptr<Kernel> kernel_init, kernel_ret, kernel_ext;

  {
    /*
    @ti.kernel
    def init():
      for index in range(n):
        place[index] = index * 2 + 1
    */
    IRBuilder builder;
    auto zero = builder.get_int32(0);
    auto n_stmt = builder.get_int32(n);
    auto loop = builder.create_range_for(zero, n_stmt, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto index = builder.get_loop_index(loop);
      auto const_2 = builder.get_int32(2);
      auto mult2 = builder.create_mul(index, const_2);
      auto const_1 = builder.get_int32(1);
      auto plus1 = builder.create_add(mult2, const_1);
      auto ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, plus1);
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
    auto sum = builder.create_local_var(PrimitiveType::i32);
    auto loop = builder.create_struct_for(pointer, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto index = builder.get_loop_index(loop);
      auto sum_old = builder.create_local_load(sum);
      auto place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    // TODO: fix this (or remove)
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");

    kernel_ret->insert_ret(PrimitiveType::i32);
    kernel_ret->finalize_rets();
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
    auto loop = builder.create_struct_for(pointer, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto index = builder.get_loop_index(loop);
      auto dt = TypeFactory::get_instance().get_ndarray_struct_type(
          get_data_type<int>(), 1);
      auto arg_load = builder.create_arg_load({0}, dt, true, 0, false);
      auto ext = builder.create_external_ptr(arg_load, {index});
      auto global_ptr_place = builder.create_global_ptr(place, {index});
      auto val = builder.create_global_load(global_ptr_place);
      builder.create_global_store(ext, val);
    }

    kernel_ext = std::make_unique<Kernel>(program, builder.extract_ir(), "ext");
    kernel_ext->insert_ndarray_param(get_data_type<int>(), /*total_dim=*/1);
    kernel_ext->finalize_params();
  }

  auto ctx_init = kernel_init->make_launch_context();
  auto ctx_ret = kernel_ret->make_launch_context();
  auto ctx_ext = kernel_ext->make_launch_context();

  auto ext_arr = std::make_unique<int[]>(n);
  ctx_ext.set_arg_external_array_with_shape({0}, (uint64)ext_arr.get(), n, {n});

  std::cout << "running init kernel ============================" << std::endl;
  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_init);
    program.launch_kernel(compiled_kernel_data, ctx_init);
  }
  std::cout << "running ret kernel ============================" << std::endl;
  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
    program.launch_kernel(compiled_kernel_data, ctx_ret);
    std::cout << "after launch ret kernel" << std::endl;
    std::cout << program.fetch_result<int>(0) << std::endl;
  }
  std::cout << "running ext kernel ============================" << std::endl;
  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_ext);
    program.launch_kernel(compiled_kernel_data, ctx_ext);
    for (int i = 0; i < n; i++)
      std::cout << ext_arr[i] << " ";
    std::cout << std::endl;
  }
}
