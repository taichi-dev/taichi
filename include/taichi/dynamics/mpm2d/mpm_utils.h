#pragma once

#include <memory>
#include <taichi/system/timer.h>
#include <taichi/common/meta.h>
#include <taichi/math/linalg.h>

TC_NAMESPACE_BEGIN

#define Pp(v) {printf("%s:\n", #v); print(v);}
#define abnormal(v) (!is_normal(v))

// Note: assuming abs(x) <= 2!!
inline real w(real x) {
    x = abs(x);
    assert(x <= 2);
    if (x < 1) {
        return 0.5f * x * x * x - x * x + 2.0f / 3.0f;
    }
    else {
        return -1.0f / 6.0f * x * x * x + x * x - 2 * x + 4.0f / 3.0f;
    }
}

// Note: assuming abs(x) <= 2!!
inline real dw(real x) {
    real s = x < 0.0f ? -1.0f : 1.0f;
    x *= s;
    assert(x <= 2.0f);
    real val;
    real xx = x * x;
    if (x < 1.0f) {
        val = 1.5f * xx - 2.0f * x;
    }
    else {
        val = -0.5f * xx + 2.0f * x - 2.0f;
    }
    return s * val;
}

inline real w(const vec2 &a) {
    return w(a.x) * w(a.y);
}

inline vec2 dw(const vec2 &a) {
    return vec2(dw(a.x) * w(a.y), w(a.x) * dw(a.y));
}

inline real det(const Matrix2 &m) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

void polar_decomp(const Matrix2 &A, Matrix2 &r, Matrix2 &s);

void svd(const Matrix2 &A, Matrix2 &u, Matrix2 &sig, Matrix2 &v);


TC_NAMESPACE_END

