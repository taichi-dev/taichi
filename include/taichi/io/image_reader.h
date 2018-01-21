/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>
#include <taichi/image/image_buffer.h>
#include <taichi/common/interface.h>

TC_NAMESPACE_BEGIN

class ImageReader : public Unit {
 public:
  ImageReader() {
  }
  virtual Array2D<Vector4> read(const std::string &filepath) {
    return Array2D<Vector4>(Vector2i(0, 0));
  }
};

TC_INTERFACE(ImageReader);

TC_NAMESPACE_END
