/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <functional>
#include "taichi/util/interface.h"
#include "pybind11/pybind11.h"
#include "taichi/util/task.h"
#include "taichi/system/benchmark.h"

TI_NAMESPACE_BEGIN

TI_INTERFACE_DEF(Benchmark, "benchmark")
TI_INTERFACE_DEF(Task, "task")

TI_NAMESPACE_END
