#include <taichi/python/export.h>

#include <taichi/common/meta.h>
#include <taichi/visualization/rgb.h>
#include <taichi/io/io.h>
#include <taichi/geometry/factory.h>

using namespace boost::python;
namespace py = boost::python;

TC_NAMESPACE_BEGIN

BOOST_PYTHON_MODULE(taichi_core) {
	Py_Initialize();
	//import_array();
	export_math();
	export_dynamics();
	export_visual();
	export_io();
	export_misc();
}

TC_NAMESPACE_END
