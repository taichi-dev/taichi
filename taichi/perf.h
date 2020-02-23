#pragma once

#include <taichi/common/util.h>
#include <taichi/system/timer.h>

TI_NAMESPACE_BEGIN

#define TI_MAX_PERF_ID 32

static inline double TI_PERF(const char *left = nullptr, int perf_id = 0, int show_freq = 1)
{
  static double sums[TI_MAX_PERF_ID];
  static int counts[TI_MAX_PERF_ID];

  static double last_time;
  double time = Time::get_time();
  double elapsed = time - last_time;
  last_time = time;

  if (left != nullptr) {
    sums[perf_id] += elapsed;
    if (++counts[perf_id] % show_freq == 0) {
      double average = sums[perf_id] / counts[perf_id];
      if (elapsed < 1e-3) {
        printf("%s ==> %6.3f us ~ %6.3f us\n", left, elapsed * 1e6, average * 1e6);
      } else {
        printf("%s ==> %6.3f ms ~ %6.3f ms\n", left, elapsed * 1e3, average * 1e3);
      }
    }
  }
  return elapsed;
}

TI_NAMESPACE_END
