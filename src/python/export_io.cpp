/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/python/export.h>
#include <taichi/io/image_reader.h>

using namespace boost::python;

EXPLICIT_GET_POINTER(taichi::ImageReader);

TC_NAMESPACE_BEGIN

void export_io() {
    def("create_image_reader", create_instance<ImageReader>);
    class_<ImageReader>("ImageReader")
        .def("initialize", &ImageReader::initialize)
        .def("read", &ImageReader::read);
    register_ptr_to_python<std::shared_ptr<ImageReader>>();
}

TC_NAMESPACE_END
