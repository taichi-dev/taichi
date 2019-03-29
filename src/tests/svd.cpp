#include "../tlang.h"
#include <taichi/testing.h>
#include <numeric>
#include <taichi/visual/gui.h>
#include <tbb/tbb.h>
#include <taichi/system/threading.h>
#include <taichi/math/svd.h>
#include "scalar_svd.h"

TLANG_NAMESPACE_BEGIN

#include "svd.h"

template <int sweeps = 5>
__attribute_noinline__ void sifakis_svd(Matrix3 &a,
                                        Matrix3 &u,
                                        Matrix3 &v,
                                        Vector3 &sig) {
  // clang-format off
  SifakisSVD::svd<sweeps>(
      a(0, 0), a(0, 1), a(0, 2),
      a(1, 0), a(1, 1), a(1, 2),
      a(2, 0), a(2, 1), a(2, 2),
      u(0, 0), u(0, 1), u(0, 2),
      u(1, 0), u(1, 1), u(1, 2),
      u(2, 0), u(2, 1), u(2, 2),
      v(0, 0), v(0, 1), v(0, 2),
      v(1, 0), v(1, 1), v(1, 2),
      v(2, 0), v(2, 1), v(2, 2),
      sig(0), sig(1), sig(2)
  );
  // clang-format on
}

int g_i = 1;
int g_j = 1;

TC_TEST("svd_benchmark") {
  using Matrix = TMatrix<float32, 3>;
  using Vector = TVector<float32, 3>;
  int N = 10000;
  auto cpe1 = measure_cpe(
      [&] {
        for (int i = 0; i < N; i++) {
          Matrix m = Matrix::rand();
          Matrix U, sig, V, Q, R, S;
          Vector sig_vec;
          sifakis_svd<4>(m, U, V, sig_vec);
          trash(U(g_i, g_j));
        }
      },
      N);
  TC_INFO("Sifakis SVD CPE: {}", cpe1);

  auto cpe2 = measure_cpe(
      [&] {
        for (int i = 0; i < N; i++) {
          Matrix m = Matrix::rand();
          Matrix U, sig, V, Q, R, S;
          taichi::svd(m, U, sig, V);
          trash(U(g_i, g_j));
        }
      },
      N);
  TC_INFO("Jixie SVD CPE: {}", cpe2);

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

TC_TEST("svd_scalar") {
  using Matrix = TMatrix<float32, 3>;
  using Vector = TVector<float32, 3>;
  float32 tolerance = 2e-3_f32;
  for (int i = 0; i < 100000; i++) {
    Matrix m = Matrix::rand();
    Matrix U, sig, V, Q, R, S;
    Vector sig_vec;

    sifakis_svd<6>(m, U, V, sig_vec);
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
}

TLANG_NAMESPACE_END
