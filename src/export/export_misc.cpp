#include <taichi/python/export.h>

#include <taichi/common/config.h>

using namespace boost::python;
namespace py = boost::python;

TC_NAMESPACE_BEGIN

Config config_from_py_dict(py::dict &c) {
	Config config;
	py::list keys = c.keys();
	for (int i = 0; i < len(keys); ++i) {
		py::object curArg = c[keys[i]];
		std::string key = py::extract<std::string>(keys[i]);
		std::string value = py::extract<std::string>(c[keys[i]]);
		config.set(key, value);
	}
	config.print_all();
	return config;
}

void test();

void export_misc() {
	def("test", test);
	def("config_from_dict", config_from_py_dict);
}

TC_NAMESPACE_END
