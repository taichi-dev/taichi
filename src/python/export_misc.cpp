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
#include <taichi/system/unit_dll.h>

EXPLICIT_GET_POINTER(taichi::ToneMapper);

TC_NAMESPACE_BEGIN

template<typename T>
void load_unit(const std::string &dll_path);

Config config_from_py_dict(py::dict &c) {
    Config config;
    for (auto item : c) {
        config.set(std::string(py::str(item.first)), std::string(py::str(item.second)));
    }
    return config;
}

void test();

void test_raise_error() {
    raise_assertion_failure_in_python("Just a test.");
}

void test_get_texture(int id) {
    auto ptr = AssetManager::get_asset<Texture>(id);
    P(ptr.use_count());
}

void print_texture_use_count(const std::shared_ptr<Texture> &tex) {
    P(tex.use_count());
}

std::shared_ptr<UnitDLL> create_unit_dll() {
    return std::make_shared<UnitDLL>();
}

void export_misc(py::module &m) {
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const ExceptionForPython &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
    
    m.def("create_tone_mapper", create_instance<ToneMapper>);
    py::class_<ToneMapper, std::shared_ptr<ToneMapper>>(m, "ToneMapper")
        .def("initialize", &ToneMapper::initialize)
        .def("apply", &ToneMapper::apply);

    m.def("create_unit_dll", create_unit_dll);
    py::class_<UnitDLL, std::shared_ptr<UnitDLL>>(m, "UnitDLL")
        .def("open_dll", &UnitDLL::open_dll)
        .def("close_dll", &UnitDLL::close_dll)
        .def("loaded", &UnitDLL::loaded);

    m.def("test", test);
    m.def("test_raise_error", test_raise_error);
    m.def("test_get_texture", test_get_texture);
    m.def("print_texture_use_count", print_texture_use_count);
    m.def("config_from_dict", config_from_py_dict);
}

TC_NAMESPACE_END
