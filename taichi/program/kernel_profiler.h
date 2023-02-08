#pragma once

#include "taichi/rhi/arch.h"
#include "taichi/util/lang_util.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <regex>

namespace taichi::lang {

struct KernelProfileTracedRecord {
  // kernel attributes
  int register_per_thread{0};
  int shared_mem_per_block{0};
  int grid_size{0};
  int block_size{0};
  int active_blocks_per_multiprocessor{0};
  // kernel time
  float kernel_elapsed_time_in_ms{0.0};
  float time_since_base{0.0};        // for Timeline
  std::string name;                  // kernel name
  std::vector<float> metric_values;  // user selected metrics
};

struct KernelProfileStatisticalResult {
  std::string name;
  int counter;
  double min;
  double max;
  double total;

  explicit KernelProfileStatisticalResult(const std::string &name)
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

  virtual void update() = 0;

  virtual bool set_profiler_toolkit(std::string toolkit_name) {
    return false;
  }

  // TODO: remove start and always use start_with_handle
  virtual void start(const std::string &kernel_name){TI_NOT_IMPLEMENTED};

  virtual TaskHandle start_with_handle(const std::string &kernel_name){
      TI_NOT_IMPLEMENTED};

  static void profiler_start(KernelProfilerBase *profiler,
                             const char *kernel_name);

  virtual void stop(){TI_NOT_IMPLEMENTED};

  virtual void stop(TaskHandle){TI_NOT_IMPLEMENTED};

  static void profiler_stop(KernelProfilerBase *profiler);

  void query(const std::string &kernel_name,
             int &counter,
             double &min,
             double &max,
             double &avg);

  std::vector<KernelProfileTracedRecord> get_traced_records() {
    return traced_records_;
  }

  double get_total_time() const;

  virtual std::string get_device_name() {
    std::string str(" ");
    return str;
  }

  virtual ~KernelProfilerBase() {
  }
};

class EventToolkitBase {
 public:
  virtual void update_record(uint32_t records_size_after_sync,
                     std::vector<KernelProfileTracedRecord> &traced_records) {
                      TI_NOT_IMPLEMENTED;
                     };
  virtual KernelProfilerBase::TaskHandle start_with_handle(
      const std::string &kernel_name) {
        TI_NOT_IMPLEMENTED;
      };
  virtual void update_timeline(std::vector<KernelProfileTracedRecord> &traced_records) {
    TI_NOT_IMPLEMENTED;
  };

 private:
  struct EventRecord {
    std::string name;
    float kernel_elapsed_time_in_ms{0.0};
    float time_since_base{0.0};
    void *start_event{nullptr};
    void *stop_event{nullptr};
  };
  float64 base_time_{0.0};
  void *base_event_{nullptr};
  // for cuEvent profiling, clear after sync()
  std::vector<EventRecord> event_records_;

 public:
  void clear() {
    event_records_.clear();
  }
  EventRecord *get_current_event_record() {
    return &(event_records_.back());
  }
  void *get_base_event() const {
    return base_event_;
  }
};

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch, bool enable);

}  // namespace taichi::lang
