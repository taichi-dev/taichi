#pragma once

#include <taichi/math/math_util.h>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

    void imp_svd(const Matrix3 &m, Matrix3 &u, Matrix3 &s, Matrix3 &v);

TC_NAMESPACE_END
