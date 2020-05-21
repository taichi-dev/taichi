#pragma once

#include "taichi/program/arch.h"
#include "taichi/lang_util.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <memory>

TLANG_NAMESPACE_BEGIN

struct ProfileRecord {
  std::string name;
  int counter;
  double min;
  double max;
  double total;

  ProfileRecord(const std::string &name)
      : name(name), counter(0), min(0), max(0), total(0) {
  }

  void insert_sample(double t);
};

class ProfilerBase {
 protected:
  std::vector<ProfileRecord> records;
  double total_time;

 public:
  void clear() {
    total_time = 0;
    records.clear();
  }

  virtual void sync() = 0;

  virtual std::string title() const = 0;

  virtual void start(const std::string &kernel_name) = 0;

  static void profiler_start(ProfilerBase *profiler, const char *kernel_name);

  virtual void stop() = 0;

  static void profiler_stop(ProfilerBase *profiler);

  void print();

  virtual ~ProfilerBase() {
  }
};

std::unique_ptr<ProfilerBase> make_profiler(Arch arch);

TLANG_NAMESPACE_END
