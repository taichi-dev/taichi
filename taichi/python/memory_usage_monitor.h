#pragma once

#include <string>
#include <fstream>
#include "taichi/common/core.h"

namespace taichi {

class MemoryMonitor {
  // avoid including py::dict
  // py::dict locals;
  void *locals_;
  std::ofstream log_;

 public:
  MemoryMonitor(int pid, std::string output_fn);
  ~MemoryMonitor();
  uint64 get_usage() const;
  void append_sample();
};

void start_memory_monitoring(std::string output_fn,
                             int pid = -1,
                             real interval = 1);

float64 get_memory_usage_gb(int pid = -1);
uint64 get_memory_usage(int pid = -1);

}  // namespace taichi
