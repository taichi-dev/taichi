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

enum class KernelProfilerMode : int {
#define PER_MODE(x) x,
#include "taichi/inc/kernel_profiler_mode.inc.h"

#undef PER_MODE
};

struct KernelProfileRecord {
  std::string name;
  int counter;
  double min;
  double max;
  double total;
  double ldg;
  double stg;
  float uti_core;
  float uti_dram;

  KernelProfileRecord(const std::string &name)
      : name(name), counter(0), min(0), max(0), total(0), ldg(0), stg(0), uti_core(0), uti_dram(0) {
  }

  void insert_sample(double t);
  void cuda_global_access(double ld,double st);
  void cuda_uti_ratio(float core,float dram);

  bool operator<(const KernelProfileRecord &o) const;
};

class KernelProfilerBase {
 protected:
  std::vector<KernelProfileRecord> records;
  double total_time_ms;
  KernelProfilerMode mode_ = KernelProfilerMode::disable;

 public:
  // Needed for the CUDA backend since we need to know which task to "stop"
  using TaskHandle = void *;

  virtual void clear();

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

  KernelProfilerMode get_mode(){return mode_;}

  virtual ~KernelProfilerBase() {
  }
};

std::unique_ptr<KernelProfilerBase> make_profiler(Arch arch, KernelProfilerMode mode);

TLANG_NAMESPACE_END
