/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

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
    export_ndarray();
}

TC_NAMESPACE_END
