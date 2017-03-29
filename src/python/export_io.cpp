/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/io/image_reader.h>

TC_NAMESPACE_BEGIN

void export_io(py::module &m) {
    py::class_<ImageReader, std::shared_ptr<ImageReader>>(m, "ImageReader")
            .def("initialize", &ImageReader::initialize)
            .def("read", &ImageReader::read);
}

TC_NAMESPACE_END
