#include "gtest/gtest.h"

#include "taichi/util/testing.h"
#include "taichi/program/async_engine.h"

namespace taichi {
namespace lang {

TEST(ParallelExecutor, CreateAndDestruct) {
  ParallelExecutor exec("test", 10);
}

TEST(ParallelExecutor, ParallelPrint) {
  int N = 100;
  std::vector<int> buffer(N, 0);
  {
    ParallelExecutor exec("test", 10);
    for (int i = 0; i < N; i++) {
      exec.enqueue([i = i, &buffer]() { buffer[i] = i + 1; });
    }
  }
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(buffer[i], i + 1);
  }
}

}  // namespace lang
}  // namespace taichi
