/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
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
