/*******************************************************************************
    Copyright (c) The Taichi Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include <taichi/util.h>
#include <taichi/testing.h>

TC_NAMESPACE_BEGIN

template <int dim, typename T>
void test_matrix() {
  using Matrix = MatrixND<dim, T>;
  T tolerance = std::is_same<T, float32>() ? 1e-5_f32 : 1e-7_f32;
  for (int i = 0; i < 1000; i++) {
    Matrix m = Matrix::rand();
    if (determinant(m) > tolerance * 1e3_f) {
      if (!math::equal(m * inversed(m), Matrix(1), tolerance)) {
        TC_P(m * inversed(m) - Matrix(1));
        TC_P(math::abs(m * inversed(m) - Matrix(1)));
        TC_P(math::maximum(math::abs(m * inversed(m) - Matrix(1))));
      }
      TC_CHECK_EQUAL(m * inversed(m), Matrix(1), tolerance);
    }
  }
}

TC_TEST("vector arith") {
  Vector3 a(1, 2, 3), b(4, 2, 5);
  CHECK(a + b == Vector3(5, 4, 8));
  CHECK(b - a == Vector3(3, 0, 2));
  CHECK(b * a == Vector3(4, 4, 15));
  CHECK(b / a == Vector3(4, 1, 5.0_f / 3.0_f));
  a += b;
  CHECK(a == Vector3(5, 4, 8));
  a -= b;
  CHECK(a == Vector3(1, 2, 3));
  a *= b;
  CHECK(a == Vector3(4, 4, 15));
  a /= b;
  CHECK(a == Vector3(1, 2, 3));
  a = Vector3(7.0_f, 8.0_f, 9.0_f);
  CHECK(a == Vector3(7, 8, 9));

  Vector2 c(1, 2), d(2, 5);
  CHECK(c + d == Vector2(3, 7));

  CHECK(Vector4(1, 2, 3, 1).length2() == 15.0_f);
#if !defined(TC_USE_DOUBLE) && !defined(TC_ISE_NONE)
  CHECK(Vector3(1, 2, 3, 1).length2() == 14.0_f);
#endif
  CHECK(dot(Vector2(1, 2), Vector2(3, 2)) == 7.0_f);
  CHECK(dot(Vector2i(1, 2), Vector2i(3, 2)) == 7);
  CHECK((fract(Vector2(1.3_f, 2.7_f)) - Vector2(0.3_f, 0.7_f)).length2() <
        1e-10_f);
  CHECK(Vector2(1.3_f, 2.7_f).sin() == Vector2(sin(1.3_f), sin(2.7_f)));

  CHECK(Matrix3(3.0_f) + Matrix3(4.0_f) == Matrix3(7.0_f));
  CHECK(Matrix3(3.0_f) + Matrix3(Vector3(1, 2, 3)) ==
        Matrix3(Vector3(4, 5, 6)));

  CHECK(Matrix2(Vector2(1, 2)) * Vector2(2, 3) == Vector2(2, 6));
  CHECK(Matrix3(Vector3(1, 2, 3)) * Vector3(2, 3, 4) == Vector3(2, 6, 12));
  CHECK(Matrix4(Vector4(1, 2, 3, 4)) * Vector4(2, 3, 4, 5) ==
        Vector4(2, 6, 12, 20));

  test_matrix<2, float32>();
  test_matrix<3, float32>();
  test_matrix<4, float32>();

  test_matrix<2, float64>();
  test_matrix<3, float64>();
  test_matrix<4, float64>();

  CHECK(math::sum(Vector4(1, 2, 3, 4)) == 10);
  CHECK(math::prod(Vector4(1, 2, 3, 4)) == 24);
  CHECK(math::sum(Vector4(1, 2, 3, 2.5)) == 8.5_f);
  CHECK(math::prod(Vector4(1, 2, 3, 2.5)) == 15);
  CHECK(math::sum(42) == 42);

  CHECK(math::exp(Vector4(1, 2, 3, 4)) ==
        Vector4(math::exp(1.0_f), math::exp(2.0_f), math::exp(3.0_f),
                math::exp(4.0_f)));

  static_assert(type::is_VectorND<VectorND<3, float>>(),
                "VectorND should be VectorND");
  static_assert(!type::is_VectorND<std::string>(),
                "std::string is not VectorND.");
}

TC_NAMESPACE_END
