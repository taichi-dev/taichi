#include "taichi/python/memory_usage_monitor.h"

#include "pybind11/embed.h"
#include "pybind11/pybind11.h"
#include "taichi/common/core.h"
#include "taichi/common/task.h"
#include "taichi/math/scalar.h"
#include "taichi/system/threading.h"
#include "taichi/system/timer.h"

TI_NAMESPACE_BEGIN

namespace py = pybind11;
using namespace py::literals;

constexpr size_t VirtualMemoryAllocator::page_size;

float64 bytes_to_GB(float64 bytes) {
  return float64(bytes) * pow<3>(1.0_f64 / 1024.0_f64);
}

float64 get_memory_usage_gb(int pid) {
  return bytes_to_GB(get_memory_usage(pid));
}

uint64 get_memory_usage(int pid) {
  if (pid == -1) {
    pid = PID::get_pid();
  }

  auto locals = py::dict("pid"_a = pid);
  py::exec(R"(
        import os, psutil
        process = psutil.Process(pid)
        mem = process.memory_info().rss)",
           py::globals(), locals);

  return locals["mem"].cast<int64>();
}

MemoryMonitor::MemoryMonitor(int pid, std::string output_fn) {
  log_.open(output_fn, std::ios_base::out);
  locals_ = new py::dict;
  (*reinterpret_cast<py::dict *>(locals_))["pid"] = pid;
  py::exec(R"(
        import os, psutil
        process = psutil.Process(pid))",
           py::globals(), *reinterpret_cast<py::dict *>(locals_));
}

MemoryMonitor::~MemoryMonitor() {
  delete reinterpret_cast<py::dict *>(locals_);
}

uint64 MemoryMonitor::get_usage() const {
  py::gil_scoped_acquire acquire;
  py::exec(R"(
        try:
          mem = process.memory_info().rss
        except:
          mem = -1)",
           py::globals(), *reinterpret_cast<py::dict *>(locals_));
  return (*reinterpret_cast<py::dict *>(locals_))["mem"].cast<uint64>();
}

void MemoryMonitor::append_sample() {
  auto t = std::chrono::system_clock::now();
  log_ << fmt::format(
      "{:.5f} {}\n",
      (t.time_since_epoch() / std::chrono::nanoseconds(1)) / 1e9_f64,
      get_usage());
  log_.flush();
}

void start_memory_monitoring(std::string output_fn, int pid, real interval) {
  if (pid == -1) {
    pid = PID::get_pid();
  }
  TI_P(pid);
  std::thread th([=]() {
    MemoryMonitor monitor(pid, output_fn);
    while (true) {
      monitor.append_sample();
      Time::sleep(interval);
    }
  });
  th.detach();
}

class MemoryTest : public Task {
 public:
  std::string run(const std::vector<std::string> &parameters) override {
    TI_P(get_memory_usage());
    Time::sleep(3);
    std::vector<uint8> a(1024ul * 1024 * 1024 * 10, 3);
    TI_P(get_memory_usage());
    Time::sleep(3);
    return "";
  }
};

class MemoryTest2 : public Task {
 public:
  std::string run(const std::vector<std::string> &parameters) override {
    start_memory_monitoring("test.txt");
    std::vector<uint8> a;
    for (int i = 0; i < 10; i++) {
      a.resize(1024ul * 1024 * 1024 * i / 2);
      std::fill(std::begin(a), std::end(a), 3);
      Time::sleep(0.5);
    }
    return "";
  }
};

TI_IMPLEMENTATION(Task, MemoryTest, "mem_test");
TI_IMPLEMENTATION(Task, MemoryTest2, "mem_test2");

TI_NAMESPACE_END
