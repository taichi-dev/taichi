/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/math.h>
#include <taichi/image/image_buffer.h>

TC_NAMESPACE_BEGIN

Array2D<Vector3> blur_with_depth(const Array2D<Vector3> &image,
                                 const Array2D<Vector3> &depth,
                                 int filter_type,
                                 real focal_plane,
                                 real aperature);

Array2D<Vector3> seam_carving(const Array2D<Vector3> &image);

TC_NAMESPACE_END
