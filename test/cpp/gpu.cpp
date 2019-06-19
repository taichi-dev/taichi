#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>

TLANG_NAMESPACE_BEGIN

TC_TEST("compiler_basics_gpu") {
  CoreState::set_trigger_gdb_when_crash(true);
  int n = 128;
  Program prog(Arch::gpu);

  Global(a, i32);
  auto i = Index(0);
  layout([&]() { root.dense(i, n).place(a); });

  auto dou = [](Expr a) { return a * 2; };

  kernel([&]() {
    For(0, n, [&](Expr i) {
      auto ret = Var(0);
      If(i % 2 == 0).Then([&] { ret = dou(i); }).Else([&] { ret = i; });
      a[i] = ret;
    });
  })();

  for (int i = 0; i < n; i++) {
    TC_CHECK(a.val<int32>(i) == (2 - i % 2) * i);
  }
};


#if defined(CUDA_FOUND)
TC_TEST("cuda_malloc_managed") {
  void *ptr;
  cudaMallocManaged(&ptr, 1LL << 40);

  int *data = (int *)ptr;
  for (int i = 0; i < 100000; i++) {
    TC_CHECK(data[i * 749] == 0);
  }
  cudaFree(ptr);
}
#endif

TLANG_NAMESPACE_END
