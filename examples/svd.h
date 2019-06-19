#pragma once
#include <taichi/lang.h>

TLANG_NAMESPACE_BEGIN
template <typename Tf = float32, typename Ti = int32>
std::tuple<Matrix, Matrix, Matrix> sifakis_svd(const Matrix &a);
TLANG_NAMESPACE_END
