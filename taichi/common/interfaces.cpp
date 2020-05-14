/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "taichi/common/interface.h"

#include <functional>
#include "pybind11/pybind11.h"

#include "taichi/common/task.h"
#include "taichi/system/benchmark.h"

TI_NAMESPACE_BEGIN

TI_INTERFACE_DEF(Benchmark, "benchmark")
TI_INTERFACE_DEF(Task, "task")

TI_NAMESPACE_END
