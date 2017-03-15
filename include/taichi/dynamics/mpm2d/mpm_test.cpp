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

TC_NAMESPACE_END
