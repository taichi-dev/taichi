#pragma once

#include "taichi/program/arch.h"
#include "taichi/lang_util.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <regex>

TLANG_NAMESPACE_BEGIN

struct KernelProfileTracedRecord {
  std::string name;
  float kernel_elapsed_time_in_ms{0.0};
  float time_since_base{0.0};        // for Timeline
  std::vector<float> metric_values;  // user selected metrics
};

struct KernelProfileStatisticalResult {
  std::string name;
  int counter;
  double min;
  double max;
  double total;

  KernelProfileStatisticalResult(const std::string &name)
      : name(name), counter(0), min(0), max(0), total(0) {
  }

  void insert_record(double t);  // TODO replace `double time` with
                                 // `KernelProfileTracedRecord record`

  bool operator<(const KernelProfileStatisticalResult &o) const;
};

class KernelProfilerBase {
 protected:
  std::vector<KernelProfileTracedRecord> traced_records_;
  std::vector<KernelProfileStatisticalResult> statistical_results_;
  double total_time_ms_{0};

 public:
  // Needed for the CUDA backend since we need to know which task to "stop"
  using TaskHandle = void *;

  virtual bool reinit_with_metrics(const std::vector<std::string> metrics) {
    return false;
  };  // public API for all backend, do not use TI_NOT_IMPLEMENTED;

  virtual void clear() = 0;

  virtual void sync() = 0;

  // TODO: remove start and always use start_with_handle
  virtual void start(const std::string &kernel_name){TI_NOT_IMPLEMENTED};

  virtual TaskHandle start_with_handle(const std::string &kernel_name){
      TI_NOT_IMPLEMENTED};

  static void profiler_start(KernelProfilerBase *profiler,
                             const char *kernel_name);

  virtual void stop(){TI_NOT_IMPLEMENTED};

  virtual void stop(TaskHandle){TI_NOT_IMPLEMENTED};

  static void profiler_stop(KernelProfilerBase *profiler);

  virtual void trace(KernelProfilerBase::TaskHandle &task_handle,
                     const std::string &task_name){TI_NOT_IMPLEMENTED};

  void query(const std::string &kernel_name,
             int &counter,
             double &min,
             double &max,
             double &avg);

  std::vector<KernelProfileTracedRecord> get_traced_records() {
    return traced_records_;
  }

  double get_total_time() const;

  virtual ~KernelProfilerBase() {
  }
};

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch, bool enable);

TLANG_NAMESPACE_END
