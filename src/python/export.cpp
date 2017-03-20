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


TC_NAMESPACE_BEGIN

PYBIND11_PLUGIN(taichi_core) {

    py::module m("taichi_core", R"pbdoc(
    Taichi Core Library
    -----------------------
    .. currentmodule:: taichi_core
    )pbdoc");

    Py_Initialize();
    export_math(m);
    export_dynamics(m);
    export_visual(m);
    export_io(m);
    export_misc(m);
    export_ndarray(m);

    return m.ptr();
}

TC_NAMESPACE_END
