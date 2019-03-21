#include "../tlang.h"
#include <taichi/system/timer.h>

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
  TC_ASSERT(f);

  CoreState::set_trigger_gdb_when_crash(true);
  Program prog;

  AmbientGlobal(x, f32, 0.0f);
  Global(y, f32);
  Global(sum, i32);

  int tree_config[] = {5, 4, 3};

  for (int i = 0; i < 3; i++) {
    tree_config[i] = 1 << tree_config[i];
  }

  layout([&] {
    auto i = Index(0), j = Index(1), k = Index(2);

    int int1_size = tree_config[0];
    int int2_size = tree_config[1];
    int leaf_size = tree_config[2];

    root.hashed({i, j, k}, int1_size)
        .fixed({i, j, k}, int2_size)
        .pointer()
        .fixed({i, j, k}, leaf_size)
        .place(x, y);

    root.place(sum);
  });

  int offset = 512;

  int num_leaves = 0;

  while (!std::feof(f)) {
    int i, j, k;

    fscanf(f, "%d%d%d", &i, &j, &k);

    i += offset;
    j += offset;
    k += offset;

    TC_ASSERT(i >= 0);
    TC_ASSERT(j >= 0);
    TC_ASSERT(k >= 0);

    num_leaves += 1;

    x.val<float32>(i, j, k) = 1;
  }

  auto count = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() { sum[Expr(0)] += 1; });
  });

  count();

  TC_P(num_leaves);
  TC_ASSERT(num_leaves * pow<3>(8) == sum.val<int>());

  auto mean_x = kernel([&] {
    Declare(i);
    Declare(j);
    Declare(k);
    For((i, j, k), x, [&]() {
      y[i, j, k] = (1.0_f / 3) * (x[i - 1, j, k] + x[i, j, k] + x[i + 1, j, k]);
    });
  });

  for (int i = 0; i < 10; i++)
    TC_TIME(mean_x());
};

TC_REGISTER_TASK(benchmark_vdb);

TLANG_NAMESPACE_END
