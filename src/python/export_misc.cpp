#include <taichi/python/export.h>
#include <taichi/python/exception.h>
#include <taichi/visual/texture.h>
#include <taichi/image/tone_mapper.h>
#include <taichi/common/asset_manager.h>
#include <taichi/math/sdf.h>
#include <taichi/system/unit_dll.h>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>

using namespace boost::python;
namespace py = boost::python;

EXPLICIT_GET_POINTER(taichi::ToneMapper);

TC_NAMESPACE_BEGIN

template<typename T>
void load_unit(const std::string &dll_path);

Config config_from_py_dict(py::dict &c) {
    Config config;
    py::list keys = c.keys();
    for (int i = 0; i < len(keys); ++i) {
        py::object curArg = c[keys[i]];
        std::string key = py::extract<std::string>(keys[i]);
        std::string value = py::extract<std::string>(c[keys[i]]);
        config.set(key, value);
    }
    return config;
}

void test();

void translate_exception_for_python(const ExceptionForPython & e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

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

void export_misc() {
    register_exception_translator<ExceptionForPython>(&translate_exception_for_python);
    
    def("create_tone_mapper", create_instance<ToneMapper>);
    class_<ToneMapper>("ToneMapper")
        .def("initialize", &ToneMapper::initialize)
        .def("apply", &ToneMapper::apply);
    register_ptr_to_python<std::shared_ptr<ToneMapper>>();
    register_ptr_to_python<std::shared_ptr<UnitDLL>>();

    def("create_unit_dll", create_unit_dll);
    class_<UnitDLL>("UnitDLL")
        .def("open_dll", &UnitDLL::open_dll)
        .def("close_dll", &UnitDLL::close_dll)
        .def("loaded", &UnitDLL::loaded)
            ;

    def("test", test);
    def("test_raise_error", test_raise_error);
    def("test_get_texture", test_get_texture);
    def("print_texture_use_count", print_texture_use_count);
    def("config_from_dict", config_from_py_dict);
}

TC_NAMESPACE_END
