#pragma once
#include "../tlang.h"

TLANG_NAMESPACE_BEGIN
std::tuple<Matrix, Matrix, Matrix> sifakis_svd(const Matrix &a);
TLANG_NAMESPACE_END
