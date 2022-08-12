/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/runtime/metal/api.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/rhi/dx/dx_api.h"
#include "taichi/common/core.h"
#include "taichi/common/interface.h"
#include "taichi/common/task.h"
#include "taichi/math/math.h"
#include "taichi/platform/cuda/detect_cuda.h"
#include "taichi/program/py_print_buffer.h"
#include "taichi/python/exception.h"
#include "taichi/python/export.h"
#include "taichi/python/memory_usage_monitor.h"
#include "taichi/system/benchmark.h"
#include "taichi/system/dynamic_loader.h"
#include "taichi/system/hacked_signal_handler.h"
#include "taichi/system/profiler.h"
#include "taichi/util/statistics.h"
#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_driver.h"
#endif

#ifdef TI_WITH_VULKAN
#include "taichi/rhi/vulkan/vulkan_loader.h"
#endif

#ifdef TI_WITH_OPENGL
#include "taichi/rhi/opengl/opengl_api.h"
#endif

#ifdef TI_WITH_CC
namespace taichi::lang::cccp {
extern bool is_c_backend_available();
}
#endif

namespace taichi {

void test_raise_error() {
  raise_assertion_failure_in_python("Just a test.");
}

void print_all_units() {
  std::vector<std::string> names;
  auto interfaces = InterfaceHolder::get_instance()->interfaces;
  for (auto &kv : interfaces) {
    names.push_back(kv.first);
  }
  std::sort(names.begin(), names.end());
  int all_units = 0;
  for (auto &interface_name : names) {
    auto impls = interfaces[interface_name]->get_implementation_names();
    std::cout << " * " << interface_name << " [" << int(impls.size()) << "]"
              << std::endl;
    all_units += int(impls.size());
    std::sort(impls.begin(), impls.end());
    for (auto &impl : impls) {
      std::cout << "   + " << impl << std::endl;
    }
  }
  std::cout << all_units << " units in all." << std::endl;
}

void export_misc(py::module &m) {
  py::class_<Config>(m, "Config");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const ExceptionForPython &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    }
  });

  py::class_<Task, std::shared_ptr<Task>>(m, "Task")
      .def("initialize", &Task::initialize)
      .def("run",
           static_cast<std::string (Task::*)(const std::vector<std::string> &)>(
               &Task::run));

  py::class_<Benchmark, std::shared_ptr<Benchmark>>(m, "Benchmark")
      .def("run", &Benchmark::run)
      .def("test", &Benchmark::test)
      .def("initialize", &Benchmark::initialize);

#define TI_EXPORT_LOGGING(X)               \
  m.def(#X, [](const std::string &msg) {   \
    taichi::Logger::get_instance().X(msg); \
  });

  m.def("flush_log", []() { taichi::Logger::get_instance().flush(); });

  TI_EXPORT_LOGGING(trace);
  TI_EXPORT_LOGGING(debug);
  TI_EXPORT_LOGGING(info);
  TI_EXPORT_LOGGING(warn);
  TI_EXPORT_LOGGING(error);
  TI_EXPORT_LOGGING(critical);

  m.def("print_all_units", print_all_units);
  m.def("set_core_state_python_imported", CoreState::set_python_imported);
  m.def("set_logging_level", [](const std::string &level) {
    Logger::get_instance().set_level(level);
  });
  m.def("logging_effective", [](const std::string &level) {
    return Logger::get_instance().is_level_effective(level);
  });
  m.def("set_logging_level_default",
        []() { Logger::get_instance().set_level_default(); });
  m.def("set_core_trigger_gdb_when_crash",
        CoreState::set_trigger_gdb_when_crash);
  m.def("test_raise_error", test_raise_error);
  m.def("get_default_float_size", []() { return sizeof(real); });
  m.def("trigger_sig_fpe", []() {
    int a = 2;
    a -= 2;
    return 1 / a;
  });
  m.def("print_profile_info",
        [&]() { Profiling::get_instance().print_profile_info(); });
  m.def("clear_profile_info",
        [&]() { Profiling::get_instance().clear_profile_info(); });
  m.def("start_memory_monitoring", start_memory_monitoring);
  m.def("get_repo_dir", get_repo_dir);
  m.def("get_python_package_dir", get_python_package_dir);
  m.def("set_python_package_dir", set_python_package_dir);
  m.def("cuda_version", get_cuda_version_string);
  m.def("test_cpp_exception", [] {
    try {
      throw std::exception();
    } catch (const std::exception &e) {
      printf("caught.\n");
    }
    printf("test was successful.\n");
  });
  m.def("pop_python_print_buffer", []() { return py_cout.pop_content(); });
  m.def("toggle_python_print_buffer", [](bool opt) { py_cout.enabled = opt; });
  m.def("with_cuda", is_cuda_api_available);
#ifdef TI_WITH_METAL
  m.def("with_metal", taichi::lang::metal::is_metal_api_available);
#else
  m.def("with_metal", []() { return false; });
#endif
#ifdef TI_WITH_OPENGL
  m.def("with_opengl", taichi::lang::opengl::is_opengl_api_available,
        py::arg("use_gles") = false);
#else
  m.def("with_opengl", []() { return false; });
#endif
#ifdef TI_WITH_VULKAN
  m.def("with_vulkan", taichi::lang::vulkan::is_vulkan_api_available);
  m.def("set_vulkan_visible_device",
        taichi::lang::vulkan::set_vulkan_visible_device);
#else
  m.def("with_vulkan", []() { return false; });
#endif
#ifdef TI_WITH_DX11
  m.def("with_dx11", taichi::lang::directx11::is_dx_api_available);
#else
  m.def("with_dx11", []() { return false; });
#endif

#ifdef TI_WITH_CC
  m.def("with_cc", taichi::lang::cccp::is_c_backend_available);
#else
  m.def("with_cc", []() { return false; });
#endif

  py::class_<Statistics>(m, "Statistics")
      .def(py::init<>())
      .def("clear", &Statistics::clear)
      .def("get_counters", &Statistics::get_counters);
  m.def(
      "get_kernel_stats", []() -> Statistics & { return stat; },
      py::return_value_policy::reference);

  py::class_<HackedSignalRegister>(m, "HackedSignalRegister").def(py::init<>());
}

}  // namespace taichi
