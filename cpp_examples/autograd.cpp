#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

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
  with ti.ad.Tape(energy):
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
      SNode *adjoint_snode() const override {
        return snode;
      }
      SNode *dual_snode() const override {
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
      SNode *adjoint_snode() const override {
        return nullptr;
      }
      SNode *dual_snode() const override {
        return nullptr;
      }
    };

    auto *snode =
        &root->dense(Axis(0), n, false).insert_children(SNodeType::place);
    snode->dt = PrimitiveType::f32;
    snode->grad_info = std::make_unique<GradInfoPrimal>(
        &root->dense(Axis(0), n, false).insert_children(SNodeType::place));
    snode->get_adjoint()->dt = PrimitiveType::f32;
    snode->get_adjoint()->grad_info = std::make_unique<GradInfoAdjoint>();
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
    auto *loop = builder.create_range_for(zero, n_stmt, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *i = builder.get_loop_index(loop);
      builder.create_global_store(builder.create_global_ptr(a, {i}), i);
      builder.create_global_store(builder.create_global_ptr(b, {i}),
                                  builder.create_add(i, one));
      builder.create_global_store(builder.create_global_ptr(c, {i}), zero);

      builder.create_global_store(
          builder.create_global_ptr(a->get_adjoint(), {i}), zero);
      builder.create_global_store(
          builder.create_global_ptr(b->get_adjoint(), {i}), zero);
      builder.create_global_store(
          builder.create_global_ptr(c->get_adjoint(), {i}), one);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  auto get_kernel_cal = [&](AutodiffMode autodiff_mode) -> Kernel * {
    IRBuilder builder;
    auto *loop = builder.create_struct_for(a, 0, 4);
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

    return new Kernel(program, builder.extract_ir(), "cal", autodiff_mode);
  };
  kernel_forward = std::unique_ptr<Kernel>(get_kernel_cal(AutodiffMode::kNone));
  kernel_backward =
      std::unique_ptr<Kernel>(get_kernel_cal(AutodiffMode::kReverse));

  {
    IRBuilder builder;
    auto *loop = builder.create_struct_for(a, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *i = builder.get_loop_index(loop);

      auto *ext_a = builder.create_external_ptr(
          builder.create_arg_load(0, PrimitiveType::f32, true), {i});
      auto *a_grad_i = builder.create_global_load(
          builder.create_global_ptr(a->get_adjoint(), {i}));
      builder.create_global_store(ext_a, a_grad_i);

      auto *ext_b = builder.create_external_ptr(
          builder.create_arg_load(1, PrimitiveType::f32, true), {i});
      auto *b_grad_i = builder.create_global_load(
          builder.create_global_ptr(b->get_adjoint(), {i}));
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
  ctx_ext.set_arg_external_array_with_shape(0, taichi::uint64(ext_a.data()), n,
                                            {n});
  ctx_ext.set_arg_external_array_with_shape(1, taichi::uint64(ext_b.data()), n,
                                            {n});
  ctx_ext.set_arg_external_array_with_shape(2, taichi::uint64(ext_c.data()), n,
                                            {n});

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
