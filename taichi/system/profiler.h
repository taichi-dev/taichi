/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "taichi/common/core.h"
#include "taichi/system/timer.h"
#include "spdlog/fmt/bundled/color.h"

TI_NAMESPACE_BEGIN

class ProfilerRecords;

// Captures running time between the construction and destruction of the
// profiler instance
class ScopedProfiler {
 public:
  ScopedProfiler(std::string name, uint64 elements = -1);

  void stop();

  static void enable();

  static void disable();

  ~ScopedProfiler();

 private:
  std::string name;
  float64 start_time;
  uint64 elements;
  bool stopped;
};

// A profiling system for multithreaded applications
class Profiling {
 public:
  void print_profile_info();
  ProfilerRecords *get_this_thread_profiler();
  static Profiling &get_instance();

 private:
  std::mutex mut;
  std::unordered_map<std::thread::id, ProfilerRecords *> profilers;
};

#define TI_PROFILER(name) taichi::ScopedProfiler _profiler_##__LINE__(name);

#define TI_AUTO_PROF TI_PROFILER(__FUNCTION__)

TI_NAMESPACE_END
