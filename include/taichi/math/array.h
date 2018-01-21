/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/math/array_1d.h>
#include <taichi/math/array_2d.h>
#include <taichi/math/array_3d.h>
#include <taichi/math/array_op.h>

TC_NAMESPACE_BEGIN
template <int DIM>
class IndexND;

template <int DIM>
class RegionND;

template <int DIM, typename T>
class ArrayND;

TC_NAMESPACE_END
