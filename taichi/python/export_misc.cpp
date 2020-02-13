/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/math/math.h>
#include <taichi/python/exception.h>
#include <taichi/python/export.h>
#include <taichi/system/benchmark.h>
#include <taichi/system/profiler.h>
#include <taichi/system/memory.h>
#include <taichi/system/unit_dll.h>
#include <taichi/geometry/factory.h>
#include <taichi/platform/metal/metal_api.h>
#if defined(TI_WITH_CUDA)
#include <cuda_runtime_api.h>
#endif

TC_NAMESPACE_BEGIN

extern Function11 python_at_exit;

Config config_from_py_dict(py::dict &c) {
  Config config;
  for (auto item : c) {
    config.set(std::string(py::str(item.first)),
               std::string(py::str(item.second)));
  }
  return config;
}

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

void duplicate_stdout_to_file(const std::string &fn) {
/*
static int stdout_fd = -1;
int fd[2];
pipe(fd);
stdout = fdopen(fd[1], "w");
auto file_fd = fdopen(fd[0], "w");
FILE *file = freopen(fn.c_str(), "w", file_fd);
*/
#if defined(TC_PLATFORM_UNIX)
  std::cerr.rdbuf(std::cout.rdbuf());
  dup2(fileno(popen(fmt::format("tee {}", fn).c_str(), "w")), STDOUT_FILENO);
#else
  TC_NOT_IMPLEMENTED;
#endif
}

void stop_duplicating_stdout_to_file(const std::string &fn) {
  TC_NOT_IMPLEMENTED;
}

bool with_cuda() {
#if defined(TI_WITH_CUDA)
  return true;
#else
  return false;
#endif
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

  py::class_<UnitDLL, std::shared_ptr<UnitDLL>>(m, "UnitDLL")
      .def("open_dll", &UnitDLL::open_dll)
      .def("close_dll", &UnitDLL::close_dll)
      .def("loaded", &UnitDLL::loaded);

#define TC_EXPORT_LOGGING(X) \
  m.def(#X, [](const std::string &msg) { taichi::logger.X(msg); });

  m.def("flush_log", []() { taichi::logger.flush(); });

  TC_EXPORT_LOGGING(trace);
  TC_EXPORT_LOGGING(debug);
  TC_EXPORT_LOGGING(info);
  TC_EXPORT_LOGGING(warn);
  TC_EXPORT_LOGGING(error);
  TC_EXPORT_LOGGING(critical);

  m.def("duplicate_stdout_to_file", duplicate_stdout_to_file);

  m.def("print_all_units", print_all_units);
  m.def("set_core_state_python_imported", CoreState::set_python_imported);
  m.def("set_logging_level",
        [](const std::string &level) { logger.set_level(level); });
  m.def("logging_effective", [](const std::string &level) {
    return logger.is_level_effective(level);
  });
  m.def("set_logging_level_default", []() { logger.set_level_default(); });
  m.def("set_core_trigger_gdb_when_crash",
        CoreState::set_trigger_gdb_when_crash);
  m.def("test_raise_error", test_raise_error);
  m.def("config_from_dict", config_from_py_dict);
  m.def("get_default_float_size", []() { return sizeof(real); });
  m.def("register_at_exit",
        [&](uint64 ptr) { python_at_exit = *(Function11 *)(ptr); });
  m.def("trigger_sig_fpe", []() {
    int a = 2;
    a -= 2;
    return 1 / a;
  });
  // m.def("dict_from_config", py_dict_from_py_config);
  m.def("print_profile_info", [&]() { print_profile_info(); });
  m.def("start_memory_monitoring", start_memory_monitoring);
  m.def("absolute_path", absolute_path);
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
  m.def("with_cuda", with_cuda);
  m.def("with_metal", taichi::Tlang::metal::is_metal_api_available);
}

TC_NAMESPACE_END
