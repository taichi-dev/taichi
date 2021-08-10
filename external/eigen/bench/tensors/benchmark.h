/*
 * Copyright (C) 2012 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stddef.h>
#include <stdint.h>
#include <vector>

namespace testing {
class Benchmark {
 public:
  Benchmark(const char* name, void (*fn)(int)) {
    Register(name, fn, NULL);
  }
  Benchmark(const char* name, void (*fn_range)(int, int)) {
    Register(name, NULL, fn_range);
  }
  Benchmark* Arg(int x);
  Benchmark* Range(int lo, int hi);
  const char* Name();
  bool ShouldRun(int argc, char* argv[]);
  void Run();
 private:
  const char* name_;
  void (*fn_)(int);
  void (*fn_range_)(int, int);
  std::vector<int> args_;
  void Register(const char* name, void (*fn)(int), void (*fn_range)(int, int));
  void RunRepeatedlyWithArg(int iterations, int arg);
  void RunWithArg(int arg);
};
}  // namespace testing
void SetBenchmarkFlopsProcessed(int64_t);
void StopBenchmarkTiming();
void StartBenchmarkTiming();
#define BENCHMARK(f) \
    static ::testing::Benchmark* _benchmark_##f __attribute__((unused)) = \
        (new ::testing::Benchmark(#f, f))
