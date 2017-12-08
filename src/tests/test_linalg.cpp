/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/testing.h>
#include <taichi/math/svd.h>
#include <taichi/math/eigen.h>

TC_NAMESPACE_BEGIN

class TestLinAlg : public Task {
  template <int DIM>
  void test_matrix_inverse() {
    using Matrix = MatrixND<DIM, real>;
    for (int i = 0; i < 100; i++) {
      Matrix M = Matrix::rand();
      TC_P(M);
      TC_P(M * inverse(M));
      assert((M * inverse(M) - Matrix(1.0_f)).frobenius_norm2() < 1e-3_f);
    }
  }

  void test_qr() {
    using Matrix = MatrixND<2, real>;
    for (int i = 0; i < 100; i++) {
      Matrix m = Matrix::rand(), q, r;
      qr_decomp(m, q, r);
      assert((m - q * r).frobenius_norm() < 1e-5_f);
      assert((q * q.transposed() - Matrix(1.0_f)).frobenius_norm() < 1e-6_f);
      assert(abs(r[0][1]) < 1e-6_f);
      assert(r[0][0] > -1e-6_f);
      assert(r[1][1] > -1e-6_f);
      TC_P(q);
      TC_P(r);
    }
  }

  void run() override {
    Vector3 a(1, 2, 3), b(4, 2, 5);
    assert(a + b == Vector3(5, 4, 8));
    assert(b - a == Vector3(3, 0, 2));
    assert(b * a == Vector3(4, 4, 15));
    assert(b / a == Vector3(4, 1, 5.0f / 3.0f));
    a += b;
    assert(a == Vector3(5, 4, 8));
    a -= b;
    assert(a == Vector3(1, 2, 3));
    a *= b;
    assert(a == Vector3(4, 4, 15));
    a /= b;
    assert(a == Vector3(1, 2, 3));
    a = Vector3(7.0f, 8.0f, 9.0f);
    assert(a == Vector3(7, 8, 9));

    Vector2 c(1, 2), d(2, 5);
    assert(c + d == Vector2(3, 7));

    assert(Vector4(1, 2, 3, 1).length2() == 15.0f);
#if !defined(TC_USE_DOUBLE) && !defined(TC_ISE_NONE)
    assert(Vector3(1, 2, 3, 1).length2() == 14.0f);
#endif
    assert(dot(Vector2(1, 2), Vector2(3, 2)) == 7.0f);
    assert(dot(Vector2i(1, 2), Vector2i(3, 2)) == 7);
    assert((fract(Vector2(1.3f, 2.7f)) - Vector2(0.3f, 0.7f)).length2() <
           1e-10f);
    assert(Vector2(1.3f, 2.7f).sin() == Vector2(sin(1.3f), sin(2.7f)));

    assert(Matrix3(3.0f) + Matrix3(4.0f) == Matrix3(7.0f));
    assert(Matrix3(3.0f) + Matrix3(Vector3(1, 2, 3)) ==
           Matrix3(Vector3(4, 5, 6)));

    assert(Matrix2(Vector2(1, 2)) * Vector2(2, 3) == Vector2(2, 6));
    assert(Matrix3(Vector3(1, 2, 3)) * Vector3(2, 3, 4) == Vector3(2, 6, 12));
    assert(Matrix4(Vector4(1, 2, 3, 4)) * Vector4(2, 3, 4, 5) ==
           Vector4(2, 6, 12, 20));

    test_matrix_inverse<2>();
    test_matrix_inverse<3>();
    test_matrix_inverse<4>();

    test_qr();

    static_assert(type::is_VectorND<VectorND<3, float>>(),
                  "VectorND should be VectorND");
    static_assert(!type::is_VectorND<std::string>(),
                  "std::string is not VectorND.");

    std::cout << "Passed." << std::endl;
  }
};

template <typename T, int dim>
void test_conversion() {
  using Vector = VectorND<dim, T>;
  using Matrix = MatrixND<dim, T>;
  Vector vec = Vector::rand();
  Matrix mat = Matrix::rand();
  CHECK(from_eigen(to_eigen(vec)) == vec);
  CHECK(from_eigen(to_eigen(mat)) == mat);
}

TC_TEST("eigen_conversion") {
  for (int T = 0; T < 1000; T++) {
    test_conversion<float32, 2>();
    test_conversion<float32, 3>();
    test_conversion<float32, 4>();
    test_conversion<float64, 2>();
    test_conversion<float64, 3>();
    test_conversion<float64, 4>();
    test_conversion<int, 2>();
    test_conversion<int, 3>();
    test_conversion<int, 4>();
  }
}

template <int dim, typename T>
inline void test_decompositions() {
  using Matrix = MatrixND<dim, T>;
  T tolerance = std::is_same<T, float32>() ? 1e-4f : 1e-4;
  for (int i = 0; i < 100; i++) {
    Matrix m = Matrix::rand();
    Matrix U, sig, V, Q, R, S;

    svd(m, U, sig, V);
    CHECK_EQUAL(m, U * sig * transposed(V), tolerance);

    if (dim == 2) {
      qr_decomp(m, Q, R);
      CHECK_EQUAL(m, Q * R, tolerance);
    }

    polar_decomp(m, R, S);
    CHECK_EQUAL(m, R * S, tolerance);
  }
};

TC_TEST("decompositions") {
  test_decompositions<2, float32>();
  test_decompositions<3, float32>();
  test_decompositions<2, float64>();
  test_decompositions<3, float64>();
}

TC_IMPLEMENTATION(Task, TestLinAlg, "test_linalg")

TC_NAMESPACE_END
