#include "mpm_utils.h"

TC_NAMESPACE_BEGIN

void polar_decomp(const Matrix2 &A, Matrix2 &r, Matrix2 &s) {
    Matrix2 u, sig, v;
    svd(A, u, sig, v);
    r = u * glm::transpose(v);
    s = v * sig * glm::transpose(v);
}

void svd(const Matrix2 &A, Matrix2 &u, Matrix2 &sig, Matrix2 &v) {
    if (frobenius_norm(A - Matrix2(1)) < 1e-6f) {
        u = sig = v = Matrix2(1);
        return;
    }
    Matrix2 AtA = glm::transpose(A) * A;
    const double a = AtA[0][0], b = AtA[0][1], c = AtA[1][0], d = AtA[1][1];
    const double D = max(0.0, (a - d) * (a - d) + 4.0 * b * c);
    const double delta = 0.5f * sqrt(D);
    const Vector2d sigma(0.5 * (a + d) + delta, 0.5 * (a + d) - delta);
    if (delta > 1e-6) {
        Vector2d v_0(b, a - sigma[0]);
        if (sgn(v_0[0]) == 0 && sgn(v_0[1]) == 0) {
            v_0 = vec2(d - sigma[0], c);
        }
        v_0 = glm::normalize(v_0);
        Vector2d v_1(AtA[1][1] - sigma[1], c);
        if (sgn(v_1[0]) == 0 && sgn(v_1[1]) == 0) {
            v_1 = vec2(b, a - sigma[1]);
        }
        v_1 = glm::normalize(v_1);
        v = Matrix2(v_0[0], v_1[0], v_0[1], v_1[1]);
        if (abnormal(v_0) || abnormal(v_1)) {
            P(A);
            P(D);
            P(sigma);
        }
    }
    else {
        v = Matrix2(1, 0, 0, 1);
    }
    sig = Matrix2(sqrt(sigma[0]), 0, 0, sqrt(sigma[1]));
    u = A * v * Matrix2(1.0f / sig[0][0], 0, 0, 1.0f / sig[1][1]);
}

TC_NAMESPACE_END
