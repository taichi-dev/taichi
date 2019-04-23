#pragma once

#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec + 1e-6 * tv.tv_usec;
}

#include "common.h"
#include "unified_allocator.h"
#include "context.h"
#include "struct.h"
#include "arithmetics.h"
#include "gpu_profiler.h"