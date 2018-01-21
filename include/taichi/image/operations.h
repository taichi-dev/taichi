/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
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
