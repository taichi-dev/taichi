#include <taichi/lang.h>
#include <taichi/testing.h>
#include <numeric>
#include <taichi/visual/gui.h>
#include <taichi/system/threading.h>
#include <taichi/math/svd.h>
#include "scalar_svd.h"
#include "svd.h"

TLANG_NAMESPACE_BEGIN

template <typename Tf, typename Ti>
Expr svd_bitwise_or(const Expr &a, const Expr &b) {
  return bit_cast<Tf>(bit_cast<Ti>(a) || bit_cast<Ti>(b));
}

template <typename Tf, typename Ti>
Expr svd_bitwise_xor(const Expr &a, const Expr &b) {
  return bit_cast<Tf>(bit_cast<Ti>(a) ^ bit_cast<Ti>(b));
}

template <typename Tf, typename Ti>
Expr svd_bitwise_and(const Expr &a, const Expr &b) {
  return bit_cast<Tf>(bit_cast<Ti>(a) && bit_cast<Ti>(b));
}

#include "svd_body.h"

template <int sweeps = 5>
void sifakis_svd_gt(Matrix3 &a,
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
  return;
  using Matrix = TMatrix<float32, 3>;
  using Vector = TVector<float32, 3>;
  int N = 10000;
  auto cpe1 = measure_cpe(
      [&] {
        for (int i = 0; i < N; i++) {
          Matrix m = Matrix::rand();
          Matrix U, sig, V, Q, R, S;
          Vector sig_vec;
          sifakis_svd_gt<4>(m, U, V, sig_vec);
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

    sifakis_svd_gt<6>(m, U, V, sig_vec);
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

TC_TEST("svd_dsl") {
  for (auto vec : {1}) {
    CoreState::set_trigger_gdb_when_crash(true);
    using TMat = TMatrix<float32, 3>;
    float32 tolerance = 2e-3_f32;

    Matrix gA(DataType::f32, 3, 3);
    Matrix gU(DataType::f32, 3, 3);
    Matrix gSigma(DataType::f32, 3, 1);
    Matrix gV(DataType::f32, 3, 3);

    // Program prog(Arch::x86_64);
    Program prog(Arch::gpu);

    constexpr int N = 2048;

    prog.layout([&] {
      auto i = Index(0);
      // TODO: SOA
      root.dense(i, N).place(gA);
      root.dense(i, N).place(gU);
      root.dense(i, N).place(gSigma);
      root.dense(i, N).place(gV);
    });

    std::vector<TMat> As;
    for (int i = 0; i < N; i++) {
      TMat A = TMat::rand();
      As.push_back(A);

      for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
          gA(p, q).val<float32>(i) = A(p, q);
        }
      }
    }

    kernel([&] {
      // Vectorize(vec);
      For(0, N, [&](Expr i) {
        auto svd = sifakis_svd<float32, int32>(gA[i]);
        gU[i] = std::get<0>(svd);
        gSigma[i] = std::get<1>(svd);
        gV[i] = std::get<2>(svd);
      });
    })();

    for (int i = 0; i < N; i++) {
      TMat m = As[i];
      TMat U, sig, V;

      for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
          U(p, q) = gU(p, q).val<float32>(i);
          V(p, q) = gV(p, q).val<float32>(i);
        }
        sig(p, p) = gSigma(p).val<float32>(i);
      }

      TC_CHECK_EQUAL(m, U * sig * transposed(V), tolerance);
      TC_CHECK_EQUAL(TMat(1), U * transposed(U), tolerance);
      TC_CHECK_EQUAL(TMat(1), V * transposed(V), tolerance);
      TC_CHECK_EQUAL(sig, TMat(sig.diag()), tolerance);
    }
  }
}

TC_TEST("svd_dsl_float64") {
  for (auto vec : {1}) {
    CoreState::set_trigger_gdb_when_crash(true);
    using TMat = TMatrix<float32, 3>;
    float32 tolerance = 2e-3_f32;

    Matrix gA(DataType::f64, 3, 3);
    Matrix gU(DataType::f64, 3, 3);
    Matrix gSigma(DataType::f64, 3, 1);
    Matrix gV(DataType::f64, 3, 3);

    // Program prog(Arch::x86_64);
    Program prog(Arch::gpu);
    prog.config.lower_access = false;

    constexpr int N = 2048;

    prog.layout([&] {
      auto i = Index(0);
      // TODO: SOA
      root.dense(i, N).place(gA);
      root.dense(i, N).place(gU);
      root.dense(i, N).place(gSigma);
      root.dense(i, N).place(gV);
    });

    std::vector<TMat> As;
    for (int i = 0; i < N; i++) {
      TMat A = TMat::rand();
      As.push_back(A);

      for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
          gA(p, q).val<float64>(i) = A(p, q);
        }
      }
    }

    kernel([&] {
      // Vectorize(vec);
      For(0, N, [&](Expr i) {
        auto svd = sifakis_svd<float64, int64>(gA[i]);
        gU[i] = std::get<0>(svd);
        gSigma[i] = std::get<1>(svd);
        gV[i] = std::get<2>(svd);
      });
    })();

    for (int i = 0; i < N; i++) {
      TMat m = As[i];
      TMat U, sig, V;

      for (int p = 0; p < 3; p++) {
        for (int q = 0; q < 3; q++) {
          U(p, q) = gU(p, q).val<float64>(i);
          V(p, q) = gV(p, q).val<float64>(i);
        }
        sig(p, p) = gSigma(p).val<float64>(i);
      }

      TC_CHECK_EQUAL(m, U * sig * transposed(V), tolerance);
      TC_CHECK_EQUAL(TMat(1), U * transposed(U), tolerance);
      TC_CHECK_EQUAL(TMat(1), V * transposed(V), tolerance);
      TC_CHECK_EQUAL(sig, TMat(sig.diag()), tolerance);
    }
  }
}

TLANG_NAMESPACE_END
