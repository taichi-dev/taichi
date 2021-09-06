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

enum class KernelProfilingMode : int {
#define PER_MODE(x) x,
#include "taichi/inc/kernel_profiling_mode.inc.h"
#undef PER_MODE
};

enum class KernelProfilingTool : int {
#define PER_TOOL(x) x,
#include "taichi/inc/kernel_profiling_tool.inc.h"
#undef PER_TOOL
};

std::string kernel_profiler_name(KernelProfilingMode mode);

KernelProfilingMode kernel_profiler_from_name(const std::string &mode);

struct KernelProfileTracedRecord {
  std::string kernel_name;
  float kernel_elapsed_time_in_ms{0.0};
#if defined(TI_WITH_TOOLKIT_CUDA)
  float kernel_gloabl_load_byets{0.0};
  float kernel_gloabl_store_byets{0.0};
  float utilization_ratio_sm{0.0};
  float utilization_ratio_mem{0.0};
#endif
};

struct KernelProfileRecord {
  std::string name;
  int counter;
  double min;
  double max;
  double total;
  float mem_load_in_bytes;
  float mem_store_in_bytes;
  float utilization_core;
  float utilization_mem;

  KernelProfileRecord(const std::string &name)
      : name(name),
        counter(0),
        min(0),
        max(0),
        total(0),
        mem_load_in_bytes(0),
        mem_store_in_bytes(0),
        utilization_core(0),
        utilization_mem(0) {
  }

  void insert_sample(double t);
  void cuda_mem_access(float load, float store);
  void cuda_utilization_ratio(float core, float mem);

  bool operator<(const KernelProfileRecord &o) const;
};

class KernelProfilerBase {
 protected:
  std::vector<KernelProfileRecord> records;
  double total_time_ms;
  KernelProfilingMode mode_ = KernelProfilingMode::disable;
  KernelProfilingTool tool_ = KernelProfilingTool::undef;

 public:
  // Needed for the CUDA backend since we need to know which task to "stop"
  using TaskHandle = void *;

  virtual void clear();
  virtual void record(KernelProfilerBase::TaskHandle &task_handle,
                      const std::string &task_name){};
  virtual void clear_toolkit(){};

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

  virtual void print();

  void query(const std::string &kernel_name,
             int &counter,
             double &min,
             double &max,
             double &avg);

  double get_total_time() const;

  KernelProfilingTool get_profiling_tool() {
    return tool_;
  }
  KernelProfilingMode get_profiling_mode() {
    return mode_;
  }

  virtual ~KernelProfilerBase() {
  }
};

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch,
                                                  KernelProfilingMode &mode);

TLANG_NAMESPACE_END
