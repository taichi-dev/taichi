#include "../tlang.h"
#include <numeric>
#include <taichi/visual/gui.h>
#include <tbb/tbb.h>
#include <taichi/system/threading.h>

void svd(float *a11,
         float *a12,
         float *a13,
         float *a21,
         float *a22,
         float *a23,
         float *a31,
         float *a32,
         float *a33,
         float *u11,
         float *u12,
         float *u13,
         float *u21,
         float *u22,
         float *u23,
         float *u31,
         float *u32,
         float *u33,
         float *v11,
         float *v12,
         float *v13,
         float *v21,
         float *v22,
         float *v23,
         float *v31,
         float *v32,
         float *v33,
         float *sigma1,
         float *sigma2,
         float *sigma3);

TLANG_NAMESPACE_BEGIN

void eftychios_svd(Matrix3 &a, Matrix3 &u, Matrix3 &v, Vector3 &sig) {
  // clang-format off
  ::svd(
      &a(0, 0), &a(0, 1), &a(0, 2),
      &a(1, 0), &a(1, 1), &a(1, 2),
      &a(2, 0), &a(2, 1), &a(2, 2),
      &u(0, 0), &u(0, 1), &u(0, 2),
      &u(1, 0), &u(1, 1), &u(1, 2),
      &u(2, 0), &u(2, 1), &u(2, 2),
      &v(0, 0), &v(0, 1), &v(0, 2),
      &v(1, 0), &v(1, 1), &v(1, 2),
      &v(2, 0), &v(2, 1), &v(2, 2),
      &sig(0), &sig(1), &sig(2)
  );
  // clang-format on
}

inline void test_decompositions() {
  constexpr int dim = 3;
  using T = float32;
  using Matrix = TMatrix<T, dim>;
  using Vector = TVector<T, dim>;
  T tolerance = std::is_same<T, float32>() ? 2e-4_f32 : 1e-12_f32;
  for (int i = 0; i < 100000; i++) {
    Matrix m = Matrix::rand();
    Matrix U, sig, V, Q, R, S;
    Vector sig_vec;

    eftychios_svd(m, U, V, sig_vec);
    sig = Matrix(sig_vec);
    TC_CHECK_EQUAL(m, U * sig * transposed(V), tolerance);
    TC_CHECK_EQUAL(Matrix(1), U * transposed(U), tolerance);
    TC_CHECK_EQUAL(Matrix(1), V * transposed(V), tolerance);
    TC_CHECK_EQUAL(sig, Matrix(sig.diag()), tolerance);

    /*
    if (dim == 2) {
      qr_decomp(m, Q, R);
      TC_CHECK_EQUAL(m, Q * R, tolerance);
      TC_CHECK_EQUAL(Q * transposed(Q), Matrix(1), tolerance);
      CHECK(abs(R[0][1]) < 1e-6_f);
      CHECK(R[0][0] > -1e-6_f);
      CHECK(R[1][1] > -1e-6_f);
    }

    polar_decomp(m, R, S);
    TC_CHECK_EQUAL(m, R * S, tolerance);
    TC_CHECK_EQUAL(Matrix(1), R * transposed(R), tolerance);
    TC_CHECK_EQUAL(S, transposed(S), tolerance);
    */
  }
};

TC_TEST("svd_scalar") {
  test_decompositions();
}

TLANG_NAMESPACE_END
