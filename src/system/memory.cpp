#include <taichi/system/memory.h>
#include <taichi/math.h>
#include <taichi/util.h>
#include <taichi/system/threading.h>
#include <taichi/system/timer.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

TC_NAMESPACE_BEGIN

constexpr size_t VirtualMemoryAllocator::page_size;

float64 get_memory_usage_gb(int pid) {
  return get_memory_usage(pid) * pow<3>(1.0_f64 / 1024.0_f64);
}

uint64 get_memory_usage(int pid) {
  if (pid == -1) {
    pid = PID::get_pid();
  }
  namespace py = pybind11;
  using namespace py::literals;

  auto locals = py::dict("pid"_a = pid);
  py::exec(R"(
        import os, psutil
        process = psutil.Process(pid)
        mem = process.memory_info().rss)",
           py::globals(), locals);

  return locals["mem"].cast<int64>();
}

class MemoryTest : public Task {
 public:
  void run(const std::vector<std::string> &parameters) override {
    TC_P(get_memory_usage());
    Time::sleep(3);
    std::vector<uint8> a(1024ul * 1024 * 1024 * 10, 3);
    TC_P(get_memory_usage());
    Time::sleep(3);
  }
};

TC_IMPLEMENTATION(Task, MemoryTest, "mem_test");

TC_NAMESPACE_END
