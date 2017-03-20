/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#pragma warning(push)
#pragma warning(disable:4244)
#pragma warning(disable:4267)

#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <vector>

#pragma warning(pop)
#define EXPLICIT_GET_POINTER(T) namespace boost {template <> T const volatile * get_pointer<T const volatile >(T const volatile *c){return c;}}
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <numpy/ndarrayobject.h>
#include <taichi/io/io.h>
#include <taichi/common/util.h>

TC_NAMESPACE_BEGIN

void export_math();

void export_dynamics();

void export_visual();

void export_misc();

void export_io();

void export_ndarray();

template<typename T>
void array2d_to_ndarray(T *arr, uint64);

template<typename T, int channels>
void image_buffer_to_ndarray(T *arr, uint64 output);

#define DEFINE_VECTOR_OF_NAMED(x, name) \
    class_<std::vector<x>>(name, init<>()) \
        .def(vector_indexing_suite<std::vector<x>, true>()) \
        .def("append", static_cast<void (std::vector<x>::*)(const x &)>(&std::vector<x>::push_back)) \
        .def("clear", &std::vector<x>::clear) \
        .def("write", &write_vector_to_disk<x>) \
        .def("read", &read_vector_from_disk<x>) \
    ;
#define DEFINE_VECTOR_OF(x) \
    class_<std::vector<x>>(#x "List", init<>()) \
        .def(vector_indexing_suite<std::vector<x>, true>()) \
        .def("append", static_cast<void (std::vector<x>::*)(const x &)>(&std::vector<x>::push_back)) \
        .def("clear", &std::vector<x>::clear) \
        .def("write", &write_vector_to_disk<x>) \
        .def("read", &read_vector_from_disk<x>) \
    ;

TC_NAMESPACE_END