#include "../tlang.h"

TLANG_NAMESPACE_BEGIN

TC_TEST("hashed_3d") {
  CoreState::set_trigger_gdb_when_crash(true);

  Program prog;

  Global(x, i32);
  Global(sum, i32);

  int n = 256;

  layout([&] {
    auto i = Index(0), j = Index(1), k = Index(2);
    root.hashed({i, j, k}, {256, 256, 256}).place(x);
    root.place(sum);
  });

  x.val<int>(2, 5, 9) = 10;
  x.val<int>(12, 5, 9) = 30;
  x.val<int>(2, 115, 9) = 11;

  auto reduce = kernel([&] {
    Declare(i);
    For(i, x, [&]() { sum[Expr(0)] += x[i]; });
  });

  reduce();

  TC_ASSERT(sum.val<int>() == 51);
};

auto benchmark_vdb = [](std::vector<std::string> param) {
  TC_ASSERT(param.size() == 1);
  auto fn = param[0];
  std::FILE *f = fopen(fn.c_str(), "r");

  CoreState::set_trigger_gdb_when_crash(true);
  Program prog;

  AmbientGlobal(x, f32, 0.0f);
  Global(y, f32);

  int tree_config[] = {5, 4, 3};

  for (int i = 0; i < 3; i++) {
    tree_config[i] = 1 << tree_config[i];
  }

  layout([&] {
    auto i = Index(0), j = Index(1), k = Index(2);

    int int1_size = tree_config[0];
    int int2_size = tree_config[1];
    int leaf_size = tree_config[2];

    root.hashed({i, j, k}, {}).fixed(i, 1024).pointer().fixed(i, 256).place(x, y);
  });

  int offset = 256;

  int num_leaves = 0;
  while (1) {
    int i, j, k;
    int ret = fscanf(f, "%d %d %d", &i, &j, &k);
    if (!ret) {
      break;
    }
    i += offset;
    j += offset;
    k += offset;

    TC_ASSERT(i >= 0);
    TC_ASSERT(j >= 0);
    TC_ASSERT(k >= 0);

    num_leaves += 1;
  }
};

TC_REGISTER_TASK(benchmark_vdb);

TLANG_NAMESPACE_END
