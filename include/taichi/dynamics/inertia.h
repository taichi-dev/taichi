/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <taichi/util.h>
#include <taichi/math/math.h>
#include <taichi/math/vector.h>

TC_NAMESPACE_BEGIN

// clang-format off

// Based on Explicit Exact Formulas for the 3-D Tetrahedron Inertia Tensor in Terms of its Vertex Coordinates by F. Tonon
// published in Journal of Mathematics and Statistics Volume 1, Issue 1 Pages 8-11
inline Matrix3 tetrahedron_inertia_tensor(const Vector3& p1, const Vector3& p2, const Vector3 & p3, const Vector3 & p4)
{
  using T = real;
  Matrix3 ret;
  T x1 = p1[0]; T x2 = p2[0]; T x3 = p3[0]; T x4 = p4[0];
  T y1 = p1[1]; T y2 = p2[1]; T y3 = p3[1]; T y4 = p4[1];
  T z1 = p1[2]; T z2 = p2[2]; T z3 = p3[2]; T z4 = p4[2];

  T det = x3*y2*z1 - x4*y2*z1 - x2*y3*z1 + x4*y3*z1 + x2*y4*z1 - x3*y4*z1 - x3*y1*z2 + x4*y1*z2 + x1*y3*z2 - x4*y3*z2 - x1*y4*z2 + x3*y4*z2 + x2*y1*z3 - x4*y1*z3 - x1*y2*z3 + x4*y2*z3 + x1*y4*z3 - x2*y4*z3 - x2*y1*z4 + x3*y1*z4 + x1*y2*z4 - x3*y2*z4 - x1*y3*z4 + x2*y3*z4;

  T a = (y1*y1 + y2*y2 + y3*y3 + y3*y4 + y4*y4 + y2*(y3 + y4) + y1*(y2 + y3 + y4) + z1*z1 + z2*z2 + z2*z3 + z3*z3 + (z2 + z3)*z4 + z4*z4 + z1*(z2 + z3 + z4)) / 60.0;
  T b = (x1*x1 + x2*x2 + x3*x3 + x3*x4 + x4*x4 + x2*(x3 + x4) + x1*(x2 + x3 + x4) + z1*z1 + z2*z2 + z2*z3 + z3*z3 + (z2 + z3)*z4 + z4*z4 + z1*(z2 + z3 + z4)) / 60.0;
  T c = (x1*x1 + x2*x2 + x3*x3 + x3*x4 + x4*x4 + x2*(x3 + x4) + x1*(x2 + x3 + x4) + y1*y1 + y2*y2 + y2*y3 + y3*y3 + (y2 + y3)*y4 + y4*y4 + y1*(y2 + y3 + y4)) / 60.0;
  T apr = (y3*z1 + y4*z1 + y3*z2 + y4*z2 + 2*y3*z3 + y4*z3 + y3*z4 + 2*y4*z4 + y1*(2*z1 + z2 + z3 + z4) + y2*(z1 + 2*z2 + z3 + z4)) / 120.0;
  T bpr = (x3*z1 + x4*z1 + x3*z2 + x4*z2 + 2*x3*z3 + x4*z3 + x3*z4 + 2*x4*z4 + x1*(2*z1 + z2 + z3 + z4) + x2*(z1 + 2*z2 + z3 + z4)) / 120.0;
  T cpr = (x3*y1 + x4*y1 + x3*y2 + x4*y2 + 2*x3*y3 + x4*y3 + x3*y4 + 2*x4*y4 + x1*(2*y1 + y2 + y3 + y4) + x2*(y1 + 2*y2 + y3 + y4)) / 120.0;

  ret[0][0] = det * a;
  ret[1][1] = det * b;
  ret[2][2] = det * c;
  ret[1][0] = - det * bpr; ret[0][1] = ret[1][0];
  ret[2][0] = - det * cpr; ret[0][2] = ret[2][0];
  ret[2][1] = - det * apr; ret[1][2] = ret[2][1];

  return -ret;
}

// clang-format on

TC_NAMESPACE_END
