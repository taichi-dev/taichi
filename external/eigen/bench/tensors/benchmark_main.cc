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
#include "benchmark.h"
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <inttypes.h>
#include <time.h>
#include <map>

static int64_t g_flops_processed;
static int64_t g_benchmark_total_time_ns;
static int64_t g_benchmark_start_time_ns;
typedef std::map<std::string, ::testing::Benchmark*> BenchmarkMap;
typedef BenchmarkMap::iterator BenchmarkMapIt;

BenchmarkMap& gBenchmarks() {
  static BenchmarkMap g_benchmarks;
  return g_benchmarks;
}

static int g_name_column_width = 20;

static int Round(int n) {
  int base = 1;
  while (base*10 < n) {
    base *= 10;
  }
  if (n < 2*base) {
    return 2*base;
  }
  if (n < 5*base) {
    return 5*base;
  }
  return 10*base;
}

#ifdef __APPLE__
  #include <mach/mach_time.h>
  static mach_timebase_info_data_t g_time_info;
  static void __attribute__((constructor)) init_info() {
    mach_timebase_info(&g_time_info);
  }
#endif

static int64_t NanoTime() {
#if defined(__APPLE__)
  uint64_t t = mach_absolute_time();
  return t * g_time_info.numer / g_time_info.denom;
#else
  struct timespec t;
  t.tv_sec = t.tv_nsec = 0;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
#endif
}

namespace testing {
Benchmark* Benchmark::Arg(int arg) {
  args_.push_back(arg);
  return this;
}

Benchmark* Benchmark::Range(int lo, int hi) {
  const int kRangeMultiplier = 8;
  if (hi < lo) {
    int temp = hi;
    hi = lo;
    lo = temp;
  }
  while (lo < hi) {
    args_.push_back(lo);
    lo *= kRangeMultiplier;
  }
  // We always run the hi number.
  args_.push_back(hi);
  return this;
}

const char* Benchmark::Name() {
  return name_;
}
bool Benchmark::ShouldRun(int argc, char* argv[]) {
  if (argc == 1) {
    return true;  // With no arguments, we run all benchmarks.
  }
  // Otherwise, we interpret each argument as a regular expression and
  // see if any of our benchmarks match.
  for (int i = 1; i < argc; i++) {
    regex_t re;
    if (regcomp(&re, argv[i], 0) != 0) {
      fprintf(stderr, "couldn't compile \"%s\" as a regular expression!\n", argv[i]);
      exit(EXIT_FAILURE);
    }
    int match = regexec(&re, name_, 0, NULL, 0);
    regfree(&re);
    if (match != REG_NOMATCH) {
      return true;
    }
  }
  return false;
}
void Benchmark::Register(const char* name, void (*fn)(int), void (*fn_range)(int, int)) {
  name_ = name;
  fn_ = fn;
  fn_range_ = fn_range;
  if (fn_ == NULL && fn_range_ == NULL) {
    fprintf(stderr, "%s: missing function\n", name_);
    exit(EXIT_FAILURE);
  }
  gBenchmarks().insert(std::make_pair(name, this));
}
void Benchmark::Run() {
  if (fn_ != NULL) {
    RunWithArg(0);
  } else {
    if (args_.empty()) {
      fprintf(stderr, "%s: no args!\n", name_);
      exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < args_.size(); ++i) {
      RunWithArg(args_[i]);
    }
  }
}
void Benchmark::RunRepeatedlyWithArg(int iterations, int arg) {
  g_flops_processed = 0;
  g_benchmark_total_time_ns = 0;
  g_benchmark_start_time_ns = NanoTime();
  if (fn_ != NULL) {
    fn_(iterations);
  } else {
    fn_range_(iterations, arg);
  }
  if (g_benchmark_start_time_ns != 0) {
    g_benchmark_total_time_ns += NanoTime() - g_benchmark_start_time_ns;
  }
}
void Benchmark::RunWithArg(int arg) {
  // run once in case it's expensive
  int iterations = 1;
  RunRepeatedlyWithArg(iterations, arg);
  while (g_benchmark_total_time_ns < 1e9 && iterations < 1e9) {
    int last = iterations;
    if (g_benchmark_total_time_ns/iterations == 0) {
      iterations = 1e9;
    } else {
      iterations = 1e9 / (g_benchmark_total_time_ns/iterations);
    }
    iterations = std::max(last + 1, std::min(iterations + iterations/2, 100*last));
    iterations = Round(iterations);
    RunRepeatedlyWithArg(iterations, arg);
  }
  char throughput[100];
  throughput[0] = '\0';
  if (g_benchmark_total_time_ns > 0 && g_flops_processed > 0) {
    double mflops_processed = static_cast<double>(g_flops_processed)/1e6;
    double seconds = static_cast<double>(g_benchmark_total_time_ns)/1e9;
    snprintf(throughput, sizeof(throughput), " %8.2f MFlops/s", mflops_processed/seconds);
  }
  char full_name[100];
  if (fn_range_ != NULL) {
    if (arg >= (1<<20)) {
      snprintf(full_name, sizeof(full_name), "%s/%dM", name_, arg/(1<<20));
    } else if (arg >= (1<<10)) {
      snprintf(full_name, sizeof(full_name), "%s/%dK", name_, arg/(1<<10));
    } else {
      snprintf(full_name, sizeof(full_name), "%s/%d", name_, arg);
    }
  } else {
    snprintf(full_name, sizeof(full_name), "%s", name_);
  }
  printf("%-*s %10d %10" PRId64 "%s\n", g_name_column_width, full_name,
         iterations, g_benchmark_total_time_ns/iterations, throughput);
  fflush(stdout);
}
}  // namespace testing
void SetBenchmarkFlopsProcessed(int64_t x) {
  g_flops_processed = x;
}
void StopBenchmarkTiming() {
  if (g_benchmark_start_time_ns != 0) {
    g_benchmark_total_time_ns += NanoTime() - g_benchmark_start_time_ns;
  }
  g_benchmark_start_time_ns = 0;
}
void StartBenchmarkTiming() {
  if (g_benchmark_start_time_ns == 0) {
    g_benchmark_start_time_ns = NanoTime();
  }
}
int main(int argc, char* argv[]) {
  if (gBenchmarks().empty()) {
    fprintf(stderr, "No benchmarks registered!\n");
    exit(EXIT_FAILURE);
  }
  for (BenchmarkMapIt it = gBenchmarks().begin(); it != gBenchmarks().end(); ++it) {
    int name_width = static_cast<int>(strlen(it->second->Name()));
    g_name_column_width = std::max(g_name_column_width, name_width);
  }
  bool need_header = true;
  for (BenchmarkMapIt it = gBenchmarks().begin(); it != gBenchmarks().end(); ++it) {
    ::testing::Benchmark* b = it->second;
    if (b->ShouldRun(argc, argv)) {
      if (need_header) {
        printf("%-*s %10s %10s\n", g_name_column_width, "", "iterations", "ns/op");
        fflush(stdout);
        need_header = false;
      }
      b->Run();
    }
  }
  if (need_header) {
    fprintf(stderr, "No matching benchmarks!\n");
    fprintf(stderr, "Available benchmarks:\n");
    for (BenchmarkMapIt it = gBenchmarks().begin(); it != gBenchmarks().end(); ++it) {
      fprintf(stderr, "  %s\n", it->second->Name());
    }
    exit(EXIT_FAILURE);
  }
  return 0;
}
