/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <functional>
#include <pybind11/pybind11.h>
#include <taichi/common/task.h>
#include <taichi/visual/texture.h>
#include <taichi/io/image_reader.h>
#include <taichi/visual/sampler.h>
#include <taichi/visual/framebuffer.h>
#include <taichi/system/benchmark.h>

TC_NAMESPACE_BEGIN

TC_INTERFACE_DEF(Texture, "texture")
TC_INTERFACE_DEF(Sampler, "sampler")
TC_INTERFACE_DEF(Framebuffer, "framebuffer")
TC_INTERFACE_DEF(Benchmark, "benchmark")
TC_INTERFACE_DEF(Task, "task")

TC_NAMESPACE_END
