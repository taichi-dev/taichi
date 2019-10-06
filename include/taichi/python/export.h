/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#endif

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <numpy/ndarrayobject.h>
#pragma GCC diagnostic pop
#include <taichi/io/io.h>
#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

namespace py = pybind11;

void export_math(py::module &m);

void export_visual(py::module &m);

void export_misc(py::module &m);

void export_io(py::module &m);

void export_ndarray(py::module &m);

#define DEFINE_VECTOR_OF_NAMED(x, name)                                   \
  py::class_<std::vector<x>>(m, name)                                     \
      .def(py::init<>())                                                  \
      .def("clear", &std::vector<x>::clear)                               \
      .def("append",                                                      \
           [](std::vector<x> &vec, const x &val) { vec.push_back(val); }) \
      .def("pop_back", &std::vector<x>::pop_back)                         \
      .def("__len__", [](const std::vector<x> &v) { return v.size(); })   \
      .def("__iter__",                                                    \
           [](std::vector<x> &v) {                                        \
             return py::make_iterator(v.begin(), v.end());                \
           },                                                             \
           py::keep_alive<0, 1>())                                        \
      .def("write", &write_vector_to_disk<x>)                             \
      .def("read", &read_vector_from_disk<x>);

#define DEFINE_VECTOR_OF(x) DEFINE_VECTOR_OF_NAMED(x, #x "List");

TC_NAMESPACE_END