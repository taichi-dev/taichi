#include "virtual_memory.h"

TI_NAMESPACE_BEGIN

class MemoryMonitor {
  // avoid including py::dict
  // py::dict locals;
  void *locals;
  std::ofstream log;

 public:
  MemoryMonitor(int pid, std::string output_fn);
  ~MemoryMonitor();
  uint64 get_usage() const;
  void append_sample();
};

void start_memory_monitoring(std::string output_fn,
                             int pid = -1,
                             real interval = 1);

TI_NAMESPACE_END
