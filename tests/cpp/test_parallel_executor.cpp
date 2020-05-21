#include "taichi/util/testing.h"
#include "taichi/program/async_engine.h"

TLANG_NAMESPACE_BEGIN

TI_TEST("parallel_executor") {
  SECTION("create_and_destruct") {
    ParallelExecutor exec(10);
  }
  SECTION("parallel_print") {
    int N = 100;
    std::vector<int> buffer(N, 0);
    {
      ParallelExecutor exec(10);
      for (int i = 0; i < N; i++) {
        exec.enqueue([i = i, &buffer]() { buffer[i] = i + 1; });
      }
    }
    for (int i = 0; i < N; i++) {
      CHECK(buffer[i] == i + 1);
    }
  }
}

TLANG_NAMESPACE_END
