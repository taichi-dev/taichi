/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/common/task.h>
#include <taichi/image/tone_mapper.h>
#include <taichi/math/math.h>
#include <taichi/math/sdf.h>
#include <taichi/python/exception.h>
#include <taichi/python/export.h>
#include <taichi/system/benchmark.h>
#include <taichi/system/profiler.h>
#include <taichi/system/unit_dll.h>
#include <taichi/visual/texture.h>
#include <taichi/geometry/factory.h>

TC_NAMESPACE_BEGIN

extern Function11 python_at_exit;

/*
py::dict py_dict_from_py_config(const Config &config) {
  py::dict d;
  for (auto key : config.get_keys()) {
    d[key] = config.get<std::string>(key);
  }
  return d;
}
*/

Config config_from_py_dict(py::dict &c) {
  Config config;
  for (auto item : c) {
    config.set(std::string(py::str(item.first)),
               std::string(py::str(item.second)));
  }
  return config;
}

void test();

void test_volumetric_io();

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

static int stdout_fd = -1;

void duplicate_stdout_to_file(const std::string &fn) {
  /*
int fd[2];
pipe(fd);
stdout = fdopen(fd[1], "w");
auto file_fd = fdopen(fd[0], "w");
FILE *file = freopen(fn.c_str(), "w", file_fd);
*/
#if !defined(_WIN64)
  std::cerr.rdbuf(std::cout.rdbuf());
  dup2(fileno(popen(fmt::format("tee {}", fn).c_str(), "w")), STDOUT_FILENO);
#endif
}

void stop_duplicating_stdout_to_file(const std::string &fn) {
  TC_NOT_IMPLEMENTED;
}

void export_misc(py::module &m) {
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
      .def("run", static_cast<void (Task::*)(const std::vector<std::string> &)>(
                      &Task::run));

  py::class_<ToneMapper, std::shared_ptr<ToneMapper>>(m, "ToneMapper")
      .def("initialize", &ToneMapper::initialize)
      .def("apply", &ToneMapper::apply);

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
  m.def("set_core_trigger_gdb_when_crash",
        CoreState::set_trigger_gdb_when_crash);
  m.def("test", test);
  m.def("test_raise_error", test_raise_error);
  m.def("test_volumetric_io", test_volumetric_io);
  m.def("config_from_dict", config_from_py_dict);
  m.def("register_at_exit",
        [&](uint64 ptr) { python_at_exit = *(Function11 *)(ptr); });
  m.def("trigger_sig_fpe", []() {
    int a = 2;
    a -= 2;
    return 1 / a;
  });
  // m.def("dict_from_config", py_dict_from_py_config);
  m.def("print_profile_info",
        [&]() { ProfilerRecords::get_instance().print(); });
}

TC_NAMESPACE_END
