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

#include "pybind11/pybind11.h"
#include "pybind11/operators.h"
#include "pybind11/stl.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "taichi/common/core.h"

TI_NAMESPACE_BEGIN

namespace py = pybind11;

void export_lang(py::module &m);

void export_math(py::module &m);

void export_misc(py::module &m);

void export_visual(py::module &m);

TI_NAMESPACE_END