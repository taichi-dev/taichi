#pragma once

#include "taichi/program/arch.h"
#include "taichi/lang_util.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <memory>

TLANG_NAMESPACE_BEGIN

struct KernelProfileRecord {
  std::string name;
  int counter;
  double min;
  double max;
  double total;

  KernelProfileRecord(const std::string &name)
      : name(name), counter(0), min(0), max(0), total(0) {
  }

  void insert_sample(double t);

  bool operator<(const KernelProfileRecord &o) const;
};

class KernelProfilerBase {
 protected:
  std::vector<KernelProfileRecord> records;
  double total_time_ms;

 public:
  // Needed for the CUDA backend since we need to know which task to "stop"
  using TaskHandle = void *;

  void clear() {
    total_time_ms = 0;
    records.clear();
  }

  virtual void sync() = 0;

  virtual std::string title() const = 0;

  // TODO: remove start and always use start_with_handle
  virtual void start(const std::string &kernel_name){TI_NOT_IMPLEMENTED};

  virtual TaskHandle start_with_handle(const std::string &kernel_name){
      TI_NOT_IMPLEMENTED};

  static void profiler_start(KernelProfilerBase *profiler,
                             const char *kernel_name);

  virtual void stop(){TI_NOT_IMPLEMENTED};

  virtual void stop(TaskHandle){TI_NOT_IMPLEMENTED};

  static void profiler_stop(KernelProfilerBase *profiler);

  void print();

  double get_total_time() const;

  virtual ~KernelProfilerBase() {
  }
};

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch);

TLANG_NAMESPACE_END
