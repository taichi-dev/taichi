/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/python/exception.h>
#include <taichi/visual/texture.h>
#include <taichi/image/tone_mapper.h>
#include <taichi/common/asset_manager.h>
#include <taichi/math/sdf.h>
#include <taichi/math/math.h>
#include <taichi/system/unit_dll.h>
#include <taichi/system/benchmark.h>
#include <taichi/system/profiler.h>
#include <taichi/common/task.h>

TC_NAMESPACE_BEGIN

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

  m.def("print_all_units", print_all_units);
  m.def("set_core_state_python_imported", CoreState::set_python_imported);
  m.def("test", test);
  m.def("test_raise_error", test_raise_error);
  m.def("test_volumetric_io", test_volumetric_io);
  m.def("config_from_dict", config_from_py_dict);
  // m.def("dict_from_config", py_dict_from_py_config);
  m.def("print_profile_info",
        [&]() { ProfilerRecords::get_instance().print(); });
}

TC_NAMESPACE_END
