/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include "mpm.h"

TC_NAMESPACE_BEGIN

void test_kernel() {
    for (real x = 1.0f; x < 2.0f; x += 0.01f) {
        real sum = w(x) + w(x - 1.0f) + w(x - 2.0f) + w(x - 3.0f);
        printf("%f\n", sum);
    }
}

void testRS() {
    Matrix2 A = Matrix2(0.6, 0.8, -0.8, 0.6) * Matrix2(2, 1, 1, 2), r, s;
    A = Matrix2(32, 31, 25, 64);
    A = Matrix2(
            0.990792, 0.195787,
            -0.192265, 0.972972);
    polar_decomp(A, r, s);
    puts("r");
    print(r);
    puts("s");
    print(s);
    puts("ident");
    print(r * glm::transpose(r));
    print(A - r * s);
}

void testSVD() {
    Matrix2 A = Matrix2(0.6, 0.8, -0.8, 0.6) * Matrix2(4, 9, -1, 2), r, s;
    Matrix2 u, sig, v;
    svd(A, u, sig, v);
    Pp(u);
    Pp(v);
    Pp(sig);
    Pp(A - u * sig * glm::transpose(v));
    Pp(u * glm::transpose(u));
    Pp(v * glm::transpose(v));
}

template <void(*T)(const Matrix2 &, Matrix2 &, Matrix2 &, Matrix2 &)>
void svd_test() {
    int test_num = 100000000;
    int error_count = 0;
    for (int k = 0; k < test_num; k++) {
        Matrix2 m;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                m[i][j] = rand() * 2 - 1;
            }
        }
        Matrix2 u, sig, v;
        T(m, u, sig, v);
        if (frobenius_norm(m - u * sig * glm::transpose(v)) > 1e-4f) {
            if (error_count < 10) {
                P(m);
                P(u);
                P(sig);
                P(v);
            }
            error_count++;
        }
    }
    printf("SVD Test error: %d / %d\n", error_count, test_num);
}

bool MPM::test() const {
    Matrix2 m(0.096664, 0.065926, 0.020765, 0.014165), r, s;
    // Matrix2 m(0.097664, 0.065926, 0.020765, 0.014165), r, s;
    // polar_decomp(m, r, s);
    // P(r);
    // P(s);
    Matrix2 u, sig, v;
    svd(m, u, sig, v);
    P(u);
    P(sig);
    P(v);
    return true;
}

TC_NAMESPACE_END
