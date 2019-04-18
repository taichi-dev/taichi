#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("gpu_gc_basics") {
  for (auto arch : {Arch::gpu}) {
    int n = 32;
    Program prog(arch);

    Global(x, i32);
    layout([&]() {
      auto i = Index(0);
      auto j = Index(1);
      root.dense(i, n).pointer().dense(j, n).place(x);
    });

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] {
        For(j, 0, i, [&] {
          Activate(x.snode(), {i, j});
        });
      });
    })();

    kernel([&]() {
      Declare(i);
      Declare(j);
      For(i, 0, n, [&] {
        For(j, 0, i, [&] {
          x[i, j] = i + j;
        });
      });
    })();

    auto stat = x.parent().parent().snode()->stat();
    TC_CHECK(stat.num_resident_blocks == n - 1);

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < i; j++) {
        TC_CHECK(x.val<int>(i, j) == i + j);
      }
    }
  }
};

TLANG_NAMESPACE_END
