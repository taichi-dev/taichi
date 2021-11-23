#include <bits/stdc++.h>

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
  auto *pointer = &root->pointer(Index(0), n);
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
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);
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
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4);
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
    auto *loop = builder.create_struct_for(pointer, 1, 0, 4);
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
  ctx_ext.set_arg_external_array(0, taichi::uint64(ext_arr.data()), n);

  (*kernel_init)(ctx_init);
  (*kernel_ret)(ctx_ret);
  std::cout << program.fetch_result<int>(0) << std::endl;
  (*kernel_ext)(ctx_ext);
  for (int i = 0; i < n; i++)
    std::cout << ext_arr[i] << " ";
  std::cout << std::endl;
}

void autograd() {
  /*
  import taichi as ti, numpy as np
  ti.init()

  n = 10
  a = ti.field(ti.f32, n, needs_grad=True)
  b = ti.field(ti.f32, n, needs_grad=True)
  c = ti.field(ti.f32, n, needs_grad=True)
  energy = ti.field(ti.f32, [], needs_grad=True)

  @ti.kernel
  def init():
      for i in range(n):
          a[i] = i
          b[i] = i + 1

  @ti.kernel
  def cal():
      for i in a:
          c[i] += a[i] + (b[i] + b[i])

  @ti.kernel
  def support(): # this function will not appear in CHI Builder code
      for i in a:
          energy += c[i]

  init()
  with ti.Tape(energy):
      cal()
      support()

  print(a.grad)
  print(b.grad)
  print(c.to_numpy())
  */
  using namespace taichi;
  using namespace lang;

  auto program = Program(Arch::x64);

  int n = 10;
  program.materialize_runtime();
  auto *root = new SNode(0, SNodeType::root);
  auto get_snode_grad = [&]() -> SNode * {
    class GradInfoPrimal final : public SNode::GradInfoProvider {
     public:
      SNode *snode;
      GradInfoPrimal(SNode *_snode) : snode(_snode) {
      }
      bool is_primal() const override {
        return true;
      }
      SNode *grad_snode() const override {
        return snode;
      }
    };
    class GradInfoAdjoint final : public SNode::GradInfoProvider {
     public:
      GradInfoAdjoint() {
      }
      bool is_primal() const override {
        return false;
      }
      SNode *grad_snode() const override {
        return nullptr;
      }
    };

    auto *snode = &root->dense(0, n).insert_children(SNodeType::place);
    snode->dt = PrimitiveType::f32;
    snode->grad_info = std::make_unique<GradInfoPrimal>(
        &root->dense(0, n).insert_children(SNodeType::place));
    snode->get_grad()->dt = PrimitiveType::f32;
    snode->get_grad()->grad_info = std::make_unique<GradInfoAdjoint>();
    return snode;
  };
  auto *a = get_snode_grad(), *b = get_snode_grad(), *c = get_snode_grad();
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/false);

  std::unique_ptr<Kernel> kernel_init, kernel_forward, kernel_backward,
      kernel_ext;

  {
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *one = builder.get_int32(1);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *i = builder.get_loop_index(loop);
      builder.create_global_store(builder.create_global_ptr(a, {i}), i);
      builder.create_global_store(builder.create_global_ptr(b, {i}),
                                  builder.create_add(i, one));
      builder.create_global_store(builder.create_global_ptr(c, {i}), zero);

      builder.create_global_store(builder.create_global_ptr(a->get_grad(), {i}),
                                  zero);
      builder.create_global_store(builder.create_global_ptr(b->get_grad(), {i}),
                                  zero);
      builder.create_global_store(builder.create_global_ptr(c->get_grad(), {i}),
                                  one);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  auto get_kernel_cal = [&](bool grad) -> Kernel * {
    IRBuilder builder;
    auto *loop = builder.create_struct_for(a, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *i = builder.get_loop_index(loop);
      auto *a_i = builder.create_global_load(builder.create_global_ptr(a, {i}));
      auto *b_i = builder.create_global_load(builder.create_global_ptr(b, {i}));
      auto *val = builder.create_add(a_i, builder.create_mul(b_i, i));
      auto *c_i = builder.create_global_ptr(c, {i});
      builder.insert(
          std::make_unique<AtomicOpStmt>(AtomicOpType::add, c_i, val));
    }

    return new Kernel(program, builder.extract_ir(), "cal", grad);
  };
  kernel_forward = std::unique_ptr<Kernel>(get_kernel_cal(false));
  kernel_backward = std::unique_ptr<Kernel>(get_kernel_cal(true));

  {
    IRBuilder builder;
    auto *loop = builder.create_struct_for(a, 1, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *i = builder.get_loop_index(loop);

      auto *ext_a = builder.create_external_ptr(
          builder.create_arg_load(0, PrimitiveType::f32, true), {i});
      auto *a_grad_i = builder.create_global_load(
          builder.create_global_ptr(a->get_grad(), {i}));
      builder.create_global_store(ext_a, a_grad_i);

      auto *ext_b = builder.create_external_ptr(
          builder.create_arg_load(1, PrimitiveType::f32, true), {i});
      auto *b_grad_i = builder.create_global_load(
          builder.create_global_ptr(b->get_grad(), {i}));
      builder.create_global_store(ext_b, b_grad_i);

      auto *ext_c = builder.create_external_ptr(
          builder.create_arg_load(2, PrimitiveType::f32, true), {i});
      auto *c_i = builder.create_global_load(builder.create_global_ptr(c, {i}));
      builder.create_global_store(ext_c, c_i);
    }

    kernel_ext = std::make_unique<Kernel>(program, builder.extract_ir(), "ext");
    kernel_ext->insert_arg(get_data_type<int>(), true);
    kernel_ext->insert_arg(get_data_type<int>(), true);
    kernel_ext->insert_arg(get_data_type<int>(), true);
  }

  auto ctx_init = kernel_init->make_launch_context();
  auto ctx_forward = kernel_forward->make_launch_context();
  auto ctx_backward = kernel_backward->make_launch_context();
  auto ctx_ext = kernel_ext->make_launch_context();
  std::vector<float> ext_a(n), ext_b(n), ext_c(n);
  ctx_ext.set_arg_external_array(0, taichi::uint64(ext_a.data()), n);
  ctx_ext.set_arg_external_array(1, taichi::uint64(ext_b.data()), n);
  ctx_ext.set_arg_external_array(2, taichi::uint64(ext_c.data()), n);

  (*kernel_init)(ctx_init);
  (*kernel_forward)(ctx_forward);
  (*kernel_backward)(ctx_backward);
  (*kernel_ext)(ctx_ext);
  for (int i = 0; i < n; i++)
    std::cout << ext_a[i] << " ";
  std::cout << std::endl;
  for (int i = 0; i < n; i++)
    std::cout << ext_b[i] << " ";
  std::cout << std::endl;
  for (int i = 0; i < n; i++)
    std::cout << ext_c[i] << " ";
  std::cout << std::endl;
}

int main() {
  run_snode();
  autograd();
  return 0;
}
