#include "tests/cpp/ir/ndarray_kernel.h"

namespace taichi::lang {

std::unique_ptr<Kernel> setup_kernel1(Program *prog) {
  IRBuilder builder1;
  {
    auto *arg = builder1.create_ndarray_arg_load(
        /*arg_id=*/{0}, get_data_type<int>(), 1, 0);
    auto *zero = builder1.get_int32(0);
    auto *one = builder1.get_int32(1);
    auto *two = builder1.get_int32(2);
    auto *a1ptr = builder1.create_external_ptr(arg, {one});
    builder1.create_global_store(a1ptr, one);  // a[1] = 1
    auto *a0 =
        builder1.create_global_load(builder1.create_external_ptr(arg, {zero}));
    auto *a2ptr = builder1.create_external_ptr(arg, {two});
    auto *a2 = builder1.create_global_load(a2ptr);
    auto *a0plusa2 = builder1.create_add(a0, a2);
    builder1.create_global_store(a2ptr, a0plusa2);  // a[2] = a[0] + a[2]
  }
  auto block = builder1.extract_ir();
  auto ker1 = std::make_unique<Kernel>(*prog, std::move(block), "ker1");
  ker1->insert_ndarray_param(get_data_type<int>(), /*total_dim=*/1);
  ker1->finalize_params();
  ker1->finalize_rets();
  return ker1;
}

std::unique_ptr<Kernel> setup_kernel2(Program *prog) {
  IRBuilder builder2;

  {
    auto *arg0 = builder2.create_ndarray_arg_load(
        /*arg_id=*/{0}, get_data_type<int>(), 1, 0);
    auto *arg1 = builder2.create_arg_load(/*arg_id=*/{1}, get_data_type<int>(),
                                          /*is_ptr=*/false, /*arg_depth=*/0);
    auto *one = builder2.get_int32(1);
    auto *a1ptr = builder2.create_external_ptr(arg0, {one});
    builder2.create_global_store(a1ptr, arg1);  // a[1] = arg1
  }
  auto block2 = builder2.extract_ir();
  auto ker2 = std::make_unique<Kernel>(*prog, std::move(block2), "ker2");
  ker2->insert_ndarray_param(get_data_type<int>(), /*total_dim=*/1);
  ker2->insert_scalar_param(get_data_type<int>());
  ker2->finalize_params();
  ker2->finalize_rets();
  return ker2;
}
}  // namespace taichi::lang
